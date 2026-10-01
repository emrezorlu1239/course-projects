"""Fine-tune openai/whisper-tiny on PolyAI/minds14 (en-US): first 450 examples train, rest eval.

Usage:
    python unit5_whisper_finetune/src/train.py --run-name lr1e-5_s500 --lr 1e-5 --max-steps 500
    python unit5_whisper_finetune/src/train.py --run-name smoke --max-steps 10 --eval-steps 10
"""

import argparse
import io
import json
from dataclasses import dataclass
from pathlib import Path

import evaluate
import librosa
import numpy as np
import soundfile as sf
import torch
from datasets import Audio, load_dataset
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    WhisperForConditionalGeneration,
    WhisperProcessor,
    set_seed,
)
from transformers.models.whisper.english_normalizer import BasicTextNormalizer

MODEL_ID = "openai/whisper-tiny"
DATASET_ID = "PolyAI/minds14"
N_TRAIN = 450
SR = 16_000
UNIT_DIR = Path(__file__).resolve().parents[1]
ROOT = UNIT_DIR.parent


def decode(audio: dict) -> np.ndarray:
    """Raw bytes -> mono float32 at 16 kHz (avoids the torchcodec dependency of datasets>=4)."""
    src = io.BytesIO(audio["bytes"]) if audio["bytes"] else audio["path"]
    y, sr = sf.read(src, dtype="float32", always_2d=True)
    y = y.mean(axis=1)
    return librosa.resample(y, orig_sr=sr, target_sr=SR) if sr != SR else y


def load_splits(processor):
    ds = load_dataset(DATASET_ID, name="en-US", split="train")
    ds = ds.cast_column("audio", Audio(decode=False))
    ds = ds.select_columns(["audio", "transcription"])

    def prepare(ex):
        y = decode(ex["audio"])
        ex["input_features"] = processor.feature_extractor(y, sampling_rate=SR).input_features[0]
        ex["input_length"] = len(y) / SR
        ex["labels"] = processor.tokenizer(ex["transcription"]).input_ids
        return ex

    ds = ds.map(prepare, remove_columns=["audio"], num_proc=1)
    train = ds.select(range(N_TRAIN))
    test = ds.select(range(N_TRAIN, len(ds)))
    # Whisper's encoder takes at most 30 s of audio.
    train = train.filter(lambda s: s < 30, input_columns=["input_length"])
    return train, test


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: WhisperProcessor
    decoder_start_token_id: int

    def __call__(self, features):
        inputs = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(inputs, return_tensors="pt")

        label_feats = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(label_feats, return_tensors="pt")
        # Padding positions -> -100 so cross-entropy ignores them.
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        # The model prepends <|startoftranscript|> itself when shifting labels right; strip it here.
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]
        batch["labels"] = labels
        return batch


def make_compute_metrics(processor):
    metric = evaluate.load("wer")
    normalizer = BasicTextNormalizer()

    def compute_metrics(pred):
        pred_ids, label_ids = pred.predictions, pred.label_ids
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
        pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = processor.batch_decode(label_ids, skip_special_tokens=True)

        wer_ortho = metric.compute(predictions=pred_str, references=label_str)

        pred_norm = [normalizer(p) for p in pred_str]
        label_norm = [normalizer(l) for l in label_str]
        # Skip references that are empty after normalisation (WER is undefined for them).
        keep = [i for i, l in enumerate(label_norm) if len(l) > 0]
        wer = metric.compute(
            predictions=[pred_norm[i] for i in keep], references=[label_norm[i] for i in keep]
        )
        # Decimal values (NOT x100), as required by the course assignment.
        return {"wer_ortho": wer_ortho, "wer": wer}

    return compute_metrics


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-name", required=True)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--max-steps", type=int, default=500)
    p.add_argument("--warmup-steps", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--eval-steps", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    set_seed(args.seed)

    processor = WhisperProcessor.from_pretrained(MODEL_ID, language="english", task="transcribe")
    model = WhisperForConditionalGeneration.from_pretrained(MODEL_ID)
    model.generation_config.language = "english"
    model.generation_config.task = "transcribe"
    model.generation_config.forced_decoder_ids = None

    train, test = load_splits(processor)
    print(f"train {len(train)} | eval {len(test)}")

    out_dir = UNIT_DIR / "outputs" / args.run_name
    targs = Seq2SeqTrainingArguments(
        output_dir=str(out_dir),
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=16,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="linear",
        warmup_steps=args.warmup_steps,
        max_steps=args.max_steps,
        fp16=True,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.eval_steps,
        save_total_limit=2,  # the best checkpoint is always retained
        logging_steps=25,
        predict_with_generate=True,
        generation_max_length=225,
        # load_best_model_at_end is unreliable in transformers 5.x (no key conversion on reload);
        # we reload the best checkpoint with from_pretrained instead.
        load_best_model_at_end=False,
        metric_for_best_model="wer",
        greater_is_better=False,
        dataloader_num_workers=0,
        report_to=["tensorboard"],
        seed=args.seed,
    )
    trainer = Seq2SeqTrainer(
        model=model,
        args=targs,
        train_dataset=train,
        eval_dataset=test,
        data_collator=DataCollatorSpeechSeq2SeqWithPadding(processor, model.config.decoder_start_token_id),
        compute_metrics=make_compute_metrics(processor),
        processing_class=processor,
    )
    baseline = trainer.evaluate()
    print("zero-shot:", baseline)
    trainer.train()

    best = trainer.state.best_model_checkpoint
    print(f"best checkpoint: {best} (wer {trainer.state.best_metric})")
    best_model = WhisperForConditionalGeneration.from_pretrained(best).to(trainer.args.device)
    best_model.generation_config = model.generation_config
    trainer.model = trainer.model_wrapped = best_model

    # Last log entry = metrics of the best checkpoint; the auto-generated model card reports it.
    final = trainer.evaluate()
    trainer.save_model()
    trainer.save_state()

    results = {
        "run_name": args.run_name,
        "model_id": MODEL_ID,
        "args": vars(args),
        "n_train": len(train),
        "n_eval": len(test),
        "zero_shot": {"wer": baseline["eval_wer"], "wer_ortho": baseline["eval_wer_ortho"]},
        "best_checkpoint": best,
        "eval_wer": final["eval_wer"],
        "eval_wer_ortho": final["eval_wer_ortho"],
        "eval_loss": final["eval_loss"],
        "history": [h for h in trainer.state.log_history if "eval_wer" in h],
    }
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / f"unit5_{args.run_name}.json").write_text(json.dumps(results, indent=2))
    print(json.dumps({k: results[k] for k in ["zero_shot", "eval_wer", "eval_wer_ortho"]}, indent=2))


if __name__ == "__main__":
    main()
