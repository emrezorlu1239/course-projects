"""Push a trained GTZAN run to the Hub with the course's required metadata.

Usage:
    python unit4_genre_classifier/src/publish.py --run-name lr3e-5_ep8 --dry-run   # writes README.md, lists files
    python unit4_genre_classifier/src/publish.py --run-name lr3e-5_ep8             # uploads
"""

import argparse
import json
import sys
from pathlib import Path

from huggingface_hub import whoami
from transformers import ASTFeatureExtractor, ASTForAudioClassification, Trainer

UNIT_DIR = Path(__file__).resolve().parents[1]
ROOT = UNIT_DIR.parent
sys.path.insert(0, str(ROOT))
from hub_card import list_upload, restore_trainer, upload, write_card  # noqa: E402

MODEL_ID = "MIT/ast-finetuned-audioset-10-10-0.4593"
MODEL_NAME = MODEL_ID.split("/")[-1]
GITHUB = "https://github.com/emrezorlu1239/course-projects/tree/main/huggingface-audio-course"


def sections(r: dict) -> dict[str, str]:
    n_tr, n_va, n_clips = r["n_train_chunks"], r["n_val_chunks"], r["n_val_clips"]
    return {
        "Model description": f"""
Audio Spectrogram Transformer (AST) pre-trained on AudioSet, fine-tuned for 10-class music genre
classification on GTZAN (Unit 4 hands-on exercise of the Hugging Face Audio Course).
The 527-class AudioSet head is replaced by a new 10-class linear head; the whole network is fine-tuned.
""",
        "Intended uses & limitations": """
Music genre classification of ~10 s, 16 kHz mono excerpts into: blues, classical, country, disco, hiphop,
jazz, metal, pop, reggae, rock. GTZAN is small (1000 x 30 s clips) and known to contain duplicates, mislabelings
and artist repetition, so accuracy on real-world music will be lower than reported here.
For a full 30 s track, average the logits of its 10 s chunks.
""",
        "Training and evaluation data": f"""
[marsyas/gtzan](https://huggingface.co/datasets/marsyas/gtzan) (999 decodable clips). Clips are resampled to
16 kHz and cut into three non-overlapping 10 s chunks (AST input: 1024 frames x 128 mel bins).
The train/validation split is stratified by genre and done **by clip** (85/15, seed 42), so chunks of one clip
never appear in both sets: {n_tr} train chunks, {n_va} validation chunks ({n_clips} clips).
Training uses SpecAugment-style time/frequency masking.

The reported accuracy is chunk-level. Averaging chunk logits per clip gives a clip-level accuracy of
{r['eval_accuracy_clip']:.4f} on the {n_clips} validation clips.

Code: [{GITHUB}]({GITHUB}) (`unit4_genre_classifier/`).
""",
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-name", required=True)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    out_dir = UNIT_DIR / "outputs" / args.run_name
    results = json.loads((ROOT / "results" / f"unit4_{args.run_name}.json").read_text())
    repo_id = f"{whoami()['name']}/{MODEL_NAME}-finetuned-gtzan"
    trainer = restore_trainer(
        Trainer,
        ASTForAudioClassification.from_pretrained(out_dir),
        ASTFeatureExtractor.from_pretrained(out_dir),
        out_dir,
        repo_id,
    )
    # Exactly the kwargs required by the course assignment (chapter4/hands_on).
    kwargs = {
        "dataset_tags": "marsyas/gtzan",
        "dataset": "GTZAN",
        "model_name": f"{MODEL_NAME}-finetuned-gtzan",
        "finetuned_from": MODEL_ID,
        "tasks": "audio-classification",
    }
    card = write_card(trainer, kwargs, sections(results), extra_tags=["audio-classification", "music-genre", "gtzan"])

    if args.dry_run:
        print(f"repo: https://huggingface.co/{repo_id} (public)\n")
        for name, mb in list_upload(out_dir):
            print(f"{name}  {mb:.1f} MB")
        print("\n" + card.read_text(encoding="utf-8"))
        return
    print(upload(trainer, "Fine-tuned AST on GTZAN"))


if __name__ == "__main__":
    main()
