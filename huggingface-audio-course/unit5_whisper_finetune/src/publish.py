"""Push a trained whisper-tiny MINDS-14 run to the Hub with the course's required metadata.

Usage:
    python unit5_whisper_finetune/src/publish.py --run-name lr1e-5_s500 --dry-run
    python unit5_whisper_finetune/src/publish.py --run-name lr1e-5_s500
"""

import argparse
import json
import sys
from pathlib import Path

from huggingface_hub import whoami
from transformers import Seq2SeqTrainer, WhisperForConditionalGeneration, WhisperProcessor

UNIT_DIR = Path(__file__).resolve().parents[1]
ROOT = UNIT_DIR.parent
sys.path.insert(0, str(ROOT))
from hub_card import list_upload, restore_trainer, upload, write_card  # noqa: E402

MODEL_ID = "openai/whisper-tiny"
REPO_NAME = "whisper-tiny-minds14-en-us"
GITHUB = "https://github.com/emrezorlu1239/course-projects/tree/main/huggingface-audio-course"


def sections(r: dict) -> dict[str, str]:
    zs = r["zero_shot"]
    return {
        "Model description": """
[openai/whisper-tiny](https://huggingface.co/openai/whisper-tiny) (39M parameters) fine-tuned for English
speech recognition on e-banking voice queries (Unit 5 hands-on exercise of the Hugging Face Audio Course).
""",
        "Intended uses & limitations": """
Transcription of short American-English customer-service / banking requests (8 kHz telephone audio,
resampled to 16 kHz). Trained on only ~450 utterances, so it is a course exercise, not a general-purpose ASR model.
""",
        "Training and evaluation data": f"""
[PolyAI/minds14](https://huggingface.co/datasets/PolyAI/minds14), `en-US` subset (563 utterances): the first 450
examples for training ({r['n_train']} kept after dropping clips longer than 30 s) and the remaining {r['n_eval']}
for evaluation.

Metrics are decimals (not percentages). `Wer` is the word error rate after Whisper's `BasicTextNormalizer`
(lower-casing, punctuation removal); `Wer Ortho` is computed on the raw text. Zero-shot whisper-tiny on the same
evaluation split: normalized WER {zs['wer']:.4f}, orthographic WER {zs['wer_ortho']:.4f}.

Code: [{GITHUB}]({GITHUB}) (`unit5_whisper_finetune/`).
""",
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-name", required=True)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    out_dir = UNIT_DIR / "outputs" / args.run_name
    results = json.loads((ROOT / "results" / f"unit5_{args.run_name}.json").read_text())
    repo_id = f"{whoami()['name']}/{REPO_NAME}"
    trainer = restore_trainer(
        Seq2SeqTrainer,
        WhisperForConditionalGeneration.from_pretrained(out_dir),
        WhisperProcessor.from_pretrained(out_dir),
        out_dir,
        repo_id,
    )
    # Required by the assignment: dataset_tags, finetuned_from, tasks (chapter5/hands_on).
    kwargs = {
        "dataset_tags": "PolyAI/minds14",
        "dataset": "MINDS-14 (en-US)",
        "language": "en",
        "model_name": "Whisper Tiny En-US - MINDS-14",
        "finetuned_from": MODEL_ID,
        "tasks": "automatic-speech-recognition",
    }
    card = write_card(trainer, kwargs, sections(results), extra_tags=["automatic-speech-recognition", "whisper"])

    if args.dry_run:
        print(f"repo: https://huggingface.co/{repo_id} (public)\n")
        for name, mb in list_upload(out_dir):
            print(f"{name}  {mb:.1f} MB")
        print("\n" + card.read_text(encoding="utf-8"))
        return
    print(upload(trainer, "Fine-tuned whisper-tiny on MINDS-14 en-US"))


if __name__ == "__main__":
    main()
