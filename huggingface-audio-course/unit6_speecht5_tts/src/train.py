"""Fine-tune microsoft/speecht5_tts on VoxPopuli Dutch (run prepare.py first).

Usage:
    python unit6_speecht5_tts/src/train.py --run-name nl_s4000 --max-steps 4000
    python unit6_speecht5_tts/src/train.py --run-name smoke --max-steps 10 --eval-steps 10
"""

import argparse
import json
from functools import partial
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from datasets import load_from_disk
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    SpeechT5ForTextToSpeech,
    SpeechT5HifiGan,
    SpeechT5Processor,
    set_seed,
)

MODEL_ID = "microsoft/speecht5_tts"
VOCODER_ID = "microsoft/speecht5_hifigan"
UNIT_DIR = Path(__file__).resolve().parents[1]
ROOT = UNIT_DIR.parent
SAMPLE_TEXTS = [
    "hallo allemaal, ik praat nederlands. groetjes aan iedereen!",
    "het europees parlement vergadert vandaag in straatsburg over het nieuwe klimaatbeleid.",
]


class TTSDataCollatorWithPadding:
    def __init__(self, processor, reduction_factor):
        self.processor = processor
        self.reduction_factor = reduction_factor

    def __call__(self, features):
        input_ids = [{"input_ids": f["input_ids"]} for f in features]
        label_features = [{"input_values": f["labels"]} for f in features]
        speaker_features = [f["speaker_embeddings"] for f in features]

        batch = self.processor.pad(input_ids=input_ids, labels=label_features, return_tensors="pt")
        # Padded spectrogram frames -> -100 so the L1 loss ignores them.
        batch["labels"] = batch["labels"].masked_fill(batch.decoder_attention_mask.unsqueeze(-1).ne(1), -100)
        del batch["decoder_attention_mask"]

        # The decoder emits `reduction_factor` frames per step, so target lengths must be multiples of it.
        if self.reduction_factor > 1:
            lengths = [len(f["input_values"]) for f in label_features]
            max_length = max(n - n % self.reduction_factor for n in lengths)
            batch["labels"] = batch["labels"][:, :max_length]

        batch["speaker_embeddings"] = torch.tensor(np.array(speaker_features), dtype=torch.float32)
        return batch


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-name", required=True)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--max-steps", type=int, default=4000)
    p.add_argument("--warmup-steps", type=int, default=500)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--eval-steps", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    set_seed(args.seed)

    ds = load_from_disk(str(UNIT_DIR / "cache" / "processed"))
    processor = SpeechT5Processor.from_pretrained(MODEL_ID)
    model = SpeechT5ForTextToSpeech.from_pretrained(MODEL_ID)
    model.config.use_cache = False  # incompatible with gradient checkpointing
    model.generate = partial(model.generate, use_cache=True)

    out_dir = UNIT_DIR / "outputs" / args.run_name
    targs = Seq2SeqTrainingArguments(
        output_dir=str(out_dir),
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_steps=args.warmup_steps,
        max_steps=args.max_steps,
        gradient_checkpointing=True,
        fp16=True,
        eval_strategy="steps",
        per_device_eval_batch_size=2,
        save_steps=args.eval_steps,
        eval_steps=args.eval_steps,
        logging_steps=25,
        save_total_limit=2,  # the best checkpoint is always retained
        # Reloaded manually below (Trainer's reload skips key conversion in transformers 5.x).
        load_best_model_at_end=False,
        metric_for_best_model="loss",
        greater_is_better=False,
        label_names=["labels"],
        dataloader_num_workers=0,
        report_to=["tensorboard"],
        seed=args.seed,
    )
    trainer = Seq2SeqTrainer(
        model=model,
        args=targs,
        train_dataset=ds["train"],
        eval_dataset=ds["test"],
        data_collator=TTSDataCollatorWithPadding(processor, model.config.reduction_factor),
        processing_class=processor,
    )
    trainer.train()

    best = trainer.state.best_model_checkpoint
    print(f"best checkpoint: {best} (eval_loss {trainer.state.best_metric})")
    best_model = SpeechT5ForTextToSpeech.from_pretrained(best).to(trainer.args.device)
    best_model.config.use_cache = True
    trainer.model = trainer.model_wrapped = best_model
    final = trainer.evaluate()
    trainer.save_model()
    trainer.save_state()

    # Listening samples: fine-tuned model + HiFi-GAN vocoder, speaker = one test-set x-vector.
    vocoder = SpeechT5HifiGan.from_pretrained(VOCODER_ID).to(trainer.args.device)
    best_model.eval()
    spk = torch.tensor(ds["test"][0]["speaker_embeddings"], dtype=torch.float32).unsqueeze(0).to(trainer.args.device)
    results_dir = ROOT / "results"
    results_dir.mkdir(exist_ok=True)
    samples = []
    for i, text in enumerate(SAMPLE_TEXTS):
        inputs = processor(text=text, return_tensors="pt").to(trainer.args.device)
        with torch.no_grad():
            speech = best_model.generate_speech(inputs["input_ids"], spk, vocoder=vocoder)
        path = results_dir / f"unit6_sample_{args.run_name}_{i}.wav"
        sf.write(path, speech.cpu().numpy(), 16000)
        samples.append({"text": text, "file": path.name, "seconds": round(len(speech) / 16000, 2)})

    results = {
        "run_name": args.run_name,
        "model_id": MODEL_ID,
        "args": vars(args),
        "n_train": len(ds["train"]),
        "n_test": len(ds["test"]),
        "best_checkpoint": best,
        "eval_loss": final["eval_loss"],
        "history": [h for h in trainer.state.log_history if "eval_loss" in h],
        "samples": samples,
    }
    (results_dir / f"unit6_{args.run_name}.json").write_text(json.dumps(results, indent=2))
    print(json.dumps({k: results[k] for k in ["eval_loss", "samples"]}, indent=2))


if __name__ == "__main__":
    main()
