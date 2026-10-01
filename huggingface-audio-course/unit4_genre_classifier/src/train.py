"""Fine-tune AST (AudioSet) on GTZAN music genres.

Usage:
    python unit4_genre_classifier/src/train.py --run-name lr3e-5 --lr 3e-5 --epochs 8
    python unit4_genre_classifier/src/train.py --run-name smoke --max-steps 10
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix
from torch.utils.data import Dataset
from transformers import ASTFeatureExtractor, ASTForAudioClassification, Trainer, TrainingArguments, set_seed

from data import build_cache

MODEL_ID = "MIT/ast-finetuned-audioset-10-10-0.4593"
UNIT_DIR = Path(__file__).resolve().parents[1]
ROOT = UNIT_DIR.parent


class ChunkDataset(Dataset):
    """Precomputed fbank chunks; optional SpecAugment (time/frequency masking) for training."""

    def __init__(self, feats, labels, augment=False, n_masks=2, max_f=24, max_t=96):
        self.feats, self.labels, self.augment = feats, labels, augment
        self.n_masks, self.max_f, self.max_t = n_masks, max_f, max_t

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        x = torch.from_numpy(self.feats[i].astype(np.float32))  # (time=1024, mel=128)
        if self.augment:
            # Normalised features have mean ~0, so masking with 0 means "replace by the dataset mean".
            for _ in range(self.n_masks):
                f = np.random.randint(0, self.max_f + 1)
                f0 = np.random.randint(0, x.shape[1] - f + 1)
                x[:, f0 : f0 + f] = 0
                t = np.random.randint(0, self.max_t + 1)
                t0 = np.random.randint(0, x.shape[0] - t + 1)
                x[t0 : t0 + t, :] = 0
        return {"input_values": x, "labels": int(self.labels[i])}


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    return {"accuracy": float((np.argmax(logits, axis=-1) == labels).mean())}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-name", required=True)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--epochs", type=float, default=8)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--grad-accum", type=int, default=2)
    p.add_argument("--warmup-ratio", type=float, default=0.1)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--label-smoothing", type=float, default=0.0)
    p.add_argument("--val-size", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-augment", action="store_true")
    p.add_argument("--grad-ckpt", action="store_true")
    p.add_argument("--max-steps", type=int, default=-1)
    args = p.parse_args()
    set_seed(args.seed)

    data = build_cache(UNIT_DIR / "cache", MODEL_ID, args.val_size, args.seed)
    names = [str(n) for n in data["label_names"]]
    tr, va = data["splits"] == "train", data["splits"] == "val"
    leak = set(data["clip_ids"][tr]) & set(data["clip_ids"][va])
    assert not leak, f"clip leakage between train and val: {sorted(leak)[:10]}"
    print(f"train chunks {tr.sum()} ({len(set(data['clip_ids'][tr]))} clips) | "
          f"val chunks {va.sum()} ({len(set(data['clip_ids'][va]))} clips)")

    train_ds = ChunkDataset(data["features"][tr], data["labels"][tr], augment=not args.no_augment)
    val_ds = ChunkDataset(data["features"][va], data["labels"][va])

    model = ASTForAudioClassification.from_pretrained(
        MODEL_ID,
        num_labels=len(names),
        id2label=dict(enumerate(names)),
        label2id={n: i for i, n in enumerate(names)},
        ignore_mismatched_sizes=True,
    )
    fe = ASTFeatureExtractor.from_pretrained(MODEL_ID)

    out_dir = UNIT_DIR / "outputs" / args.run_name
    targs = TrainingArguments(
        output_dir=str(out_dir),
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        gradient_accumulation_steps=args.grad_accum,
        gradient_checkpointing=args.grad_ckpt,
        warmup_steps=args.warmup_ratio,  # float in [0, 1) = ratio of total steps
        weight_decay=args.weight_decay,
        label_smoothing_factor=args.label_smoothing,
        lr_scheduler_type="cosine",
        fp16=True,
        eval_strategy="steps" if args.max_steps > 0 else "epoch",
        save_strategy="steps" if args.max_steps > 0 else "epoch",
        eval_steps=args.max_steps if args.max_steps > 0 else None,
        save_steps=args.max_steps if args.max_steps > 0 else None,
        logging_steps=10,
        # load_best_model_at_end is broken for AST in transformers 5.x (raw state_dict load
        # without key conversion -> nothing is loaded). We reload the best checkpoint ourselves.
        load_best_model_at_end=False,
        metric_for_best_model="accuracy",
        save_total_limit=2,  # the best checkpoint is always retained
        dataloader_num_workers=0,
        report_to=["tensorboard"],
        seed=args.seed,
    )
    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=fe,
        compute_metrics=compute_metrics,
    )
    trainer.train()

    best = trainer.state.best_model_checkpoint
    print(f"best checkpoint: {best} (metric {trainer.state.best_metric})")
    trainer.model = ASTForAudioClassification.from_pretrained(best).to(trainer.args.device)
    trainer.model_wrapped = trainer.model

    # Final evaluation of the best checkpoint: this becomes the last log entry,
    # which is what the auto-generated model card reports.
    final = trainer.evaluate()
    trainer.save_model()
    trainer.save_state()

    pred = trainer.predict(val_ds)
    logits, labels = pred.predictions, pred.label_ids
    chunk_pred = logits.argmax(-1)
    # Clip-level accuracy: average chunk logits per clip.
    clips = data["clip_ids"][va]
    clip_correct = []
    for c in np.unique(clips):
        m = clips == c
        clip_correct.append(logits[m].mean(0).argmax() == labels[m][0])

    cm = confusion_matrix(labels, chunk_pred, labels=range(len(names)))
    fig, ax = plt.subplots(figsize=(8, 7))
    ConfusionMatrixDisplay(cm, display_labels=names).plot(ax=ax, xticks_rotation=45, colorbar=False)
    ax.set_title(f"GTZAN val (chunk level), acc={final['eval_accuracy']:.4f}")
    fig.tight_layout()
    results_dir = ROOT / "results"
    results_dir.mkdir(exist_ok=True)
    fig.savefig(results_dir / f"unit4_confusion_{args.run_name}.png", dpi=120)

    results = {
        "run_name": args.run_name,
        "model_id": MODEL_ID,
        "args": vars(args),
        "n_train_chunks": int(tr.sum()),
        "n_val_chunks": int(va.sum()),
        "n_val_clips": int(len(np.unique(clips))),
        "eval_accuracy_chunk": final["eval_accuracy"],
        "eval_loss": final["eval_loss"],
        "eval_accuracy_clip": float(np.mean(clip_correct)),
        "best_checkpoint": trainer.state.best_model_checkpoint,
        "confusion_matrix": cm.tolist(),
        "label_names": names,
        "classification_report": classification_report(
            labels, chunk_pred, target_names=names, output_dict=True, zero_division=0
        ),
    }
    (results_dir / f"unit4_{args.run_name}.json").write_text(json.dumps(results, indent=2))
    print(json.dumps({k: results[k] for k in ["eval_accuracy_chunk", "eval_accuracy_clip", "eval_loss"]}, indent=2))


if __name__ == "__main__":
    main()
