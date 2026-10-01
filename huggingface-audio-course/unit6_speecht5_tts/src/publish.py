"""Push a trained SpeechT5 VoxPopuli-nl run to the Hub, tagged as text-to-speech.

Usage:
    python unit6_speecht5_tts/src/publish.py --run-name nl_s4000 --dry-run
    python unit6_speecht5_tts/src/publish.py --run-name nl_s4000
"""

import argparse
import json
import sys
from pathlib import Path

from huggingface_hub import whoami
from transformers import Seq2SeqTrainer, SpeechT5ForTextToSpeech, SpeechT5Processor

UNIT_DIR = Path(__file__).resolve().parents[1]
ROOT = UNIT_DIR.parent
sys.path.insert(0, str(ROOT))
from hub_card import list_upload, restore_trainer, upload, write_card  # noqa: E402

MODEL_ID = "microsoft/speecht5_tts"
REPO_NAME = "speecht5_finetuned_voxpopuli_nl"
GITHUB = "https://github.com/emrezorlu1239/course-projects/tree/main/huggingface-audio-course"

USAGE = """
```python
import torch
from transformers import SpeechT5ForTextToSpeech, SpeechT5HifiGan, SpeechT5Processor

repo = "{repo_id}"
processor = SpeechT5Processor.from_pretrained(repo)
model = SpeechT5ForTextToSpeech.from_pretrained(repo)
vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")

# 512-d x-vector of the target speaker (speechbrain/spkrec-xvect-voxceleb, L2-normalised)
speaker_embeddings = torch.zeros(1, 512)  # replace with a real x-vector
inputs = processor(text="hallo allemaal, ik praat nederlands.", return_tensors="pt")
speech = model.generate_speech(inputs["input_ids"], speaker_embeddings, vocoder=vocoder)  # 16 kHz waveform
```
"""


def sections(r: dict, stats: dict, repo_id: str) -> dict[str, str]:
    return {
        "Model description": f"""
[microsoft/speecht5_tts](https://huggingface.co/microsoft/speecht5_tts) fine-tuned for **Dutch** text-to-speech
(Unit 6 hands-on exercise of the Hugging Face Audio Course). The model predicts 80-bin log-mel spectrograms
conditioned on a 512-d x-vector speaker embedding; a HiFi-GAN vocoder
([microsoft/speecht5_hifigan](https://huggingface.co/microsoft/speecht5_hifigan)) turns them into 16 kHz audio.
{USAGE.format(repo_id=repo_id)}""",
        "Intended uses & limitations": """
Dutch speech synthesis for experimentation. The SpeechT5 tokenizer is character-based and English: Dutch accented
characters are mapped to their base letters (e.g. ë -> e) and numbers must be written out as words.
Voices come from European Parliament speeches; do not use the model to imitate real people.
""",
        "Training and evaluation data": f"""
[facebook/voxpopuli](https://huggingface.co/datasets/facebook/voxpopuli), `nl` train split (European Parliament
speeches). Only speakers with 100-400 utterances are kept ({stats['n_kept_speakers']} of {stats['n_raw_speakers']}
speakers), inputs with >= 200 tokens are dropped, and the rest is split 90/10:
{r['n_train']} training and {r['n_test']} evaluation utterances. Speaker embeddings are x-vectors from
[speechbrain/spkrec-xvect-voxceleb](https://huggingface.co/speechbrain/spkrec-xvect-voxceleb).
The training loss is L1 on the mel spectrogram (before and after the post-net) + binary cross-entropy on the
stop token + guided attention loss on the cross-attention.

Code: [{GITHUB}]({GITHUB}) (`unit6_speecht5_tts/`).
""",
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-name", required=True)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    out_dir = UNIT_DIR / "outputs" / args.run_name
    results = json.loads((ROOT / "results" / f"unit6_{args.run_name}.json").read_text())
    stats = json.loads((UNIT_DIR / "cache" / "prepare_stats.json").read_text())
    repo_id = f"{whoami()['name']}/{REPO_NAME}"
    trainer = restore_trainer(
        Seq2SeqTrainer,
        SpeechT5ForTextToSpeech.from_pretrained(out_dir),
        SpeechT5Processor.from_pretrained(out_dir),
        out_dir,
        repo_id,
    )
    kwargs = {
        "dataset_tags": "facebook/voxpopuli",
        "dataset": "VoxPopuli (nl)",
        "language": "nl",
        "model_name": "SpeechT5 TTS Dutch (VoxPopuli)",
        "finetuned_from": MODEL_ID,
        "tasks": "text-to-speech",
    }
    card = write_card(trainer, kwargs, sections(results, stats, repo_id), extra_tags=["text-to-speech", "speecht5"])
    # Make the Hub list it under the text-to-speech task (what the course progress checker filters on).
    text = card.read_text(encoding="utf-8")
    if "pipeline_tag:" not in text:
        text = text.replace("library_name: transformers\n", "library_name: transformers\npipeline_tag: text-to-speech\n", 1)
        card.write_text(text, encoding="utf-8")

    if args.dry_run:
        print(f"repo: https://huggingface.co/{repo_id} (public)\n")
        for name, mb in list_upload(out_dir):
            print(f"{name}  {mb:.1f} MB")
        print("\n" + card.read_text(encoding="utf-8"))
        return
    print(upload(trainer, "Fine-tuned SpeechT5 on VoxPopuli Dutch"))


if __name__ == "__main__":
    main()
