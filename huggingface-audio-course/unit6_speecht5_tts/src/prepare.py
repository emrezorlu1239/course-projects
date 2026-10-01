"""Prepare VoxPopuli Dutch for SpeechT5 TTS fine-tuning (Hugging Face Audio Course, Unit 6).

Steps: keep speakers with 100-400 utterances, clean characters the (English) SpeechT5
tokenizer does not know, compute log-mel targets with the SpeechT5 processor, compute a
512-d x-vector speaker embedding per utterance (SpeechBrain), drop inputs with >= 200 tokens.
The processed dataset is saved to unit6_speecht5_tts/cache/processed.

Usage: python unit6_speecht5_tts/src/prepare.py
"""

import io
import json
from collections import Counter
from glob import glob
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch
from datasets import Audio, load_dataset
from huggingface_hub import snapshot_download
from speechbrain.inference.speaker import EncoderClassifier
from transformers import SpeechT5Processor

MODEL_ID = "microsoft/speecht5_tts"
SPK_MODEL_ID = "speechbrain/spkrec-xvect-voxceleb"
SR = 16_000
UNIT_DIR = Path(__file__).resolve().parents[1]
CACHE = UNIT_DIR / "cache"

# Accented characters missing from the SpeechT5 vocabulary (same mapping as the course).
REPLACEMENTS = [("à", "a"), ("ç", "c"), ("è", "e"), ("ë", "e"), ("í", "i"), ("ï", "i"), ("ö", "o"), ("ü", "u")]


def decode(audio: dict) -> np.ndarray:
    y, sr = sf.read(io.BytesIO(audio["bytes"]), dtype="float32", always_2d=True)
    y = y.mean(axis=1)
    return librosa.resample(y, orig_sr=sr, target_sr=SR) if sr != SR else y


def cleanup(text: str) -> str:
    for src, dst in REPLACEMENTS:
        text = text.replace(src, dst)
    return text


def main():
    local = snapshot_download("facebook/voxpopuli", repo_type="dataset", allow_patterns=["nl/train-*"])
    files = sorted(glob(str(Path(local) / "nl" / "train-*.parquet")))
    ds = load_dataset("parquet", data_files=files, split="train")
    ds = ds.cast_column("audio", Audio(decode=False))
    print("raw:", len(ds))

    counts = Counter(ds["speaker_id"])
    keep = {s for s, n in counts.items() if 100 <= n <= 400}
    ds = ds.filter(lambda s: s in keep, input_columns=["speaker_id"])
    print(f"speakers with 100-400 utterances: {len(keep)} -> {len(ds)} examples")

    processor = SpeechT5Processor.from_pretrained(MODEL_ID)
    vocab = set(processor.tokenizer.get_vocab())
    ds = ds.map(lambda t: {"normalized_text": cleanup(t)}, input_columns=["normalized_text"])
    unknown = Counter(c for t in ds["normalized_text"] for c in t if c != " " and c not in vocab)
    print("characters still unknown to the tokenizer:", dict(unknown))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    spk_model = EncoderClassifier.from_hparams(
        source=SPK_MODEL_ID, savedir=str(CACHE / "spkrec-xvect-voxceleb"), run_opts={"device": device}
    )

    def prepare(ex):
        y = decode(ex["audio"])
        out = processor(text=ex["normalized_text"], audio_target=y, sampling_rate=SR, return_attention_mask=False)
        out["labels"] = out["labels"][0]  # (frames, 80) log-mel target
        with torch.no_grad():
            emb = spk_model.encode_batch(torch.tensor(y, device=device).unsqueeze(0))
            emb = torch.nn.functional.normalize(emb, dim=2)
        out["speaker_embeddings"] = emb.squeeze().cpu().numpy()
        return out

    ds = ds.map(prepare, remove_columns=ds.column_names, num_proc=1, desc="features")
    ds = ds.filter(lambda ids: len(ids) < 200, input_columns=["input_ids"])
    ds = ds.train_test_split(test_size=0.1, seed=42)
    print(ds)
    ds.save_to_disk(str(CACHE / "processed"))
    stats = {
        "n_raw_speakers": len(counts),
        "n_kept_speakers": len(keep),
        "n_train": len(ds["train"]),
        "n_test": len(ds["test"]),
        "unknown_chars_after_cleanup": dict(unknown),
    }
    (CACHE / "prepare_stats.json").write_text(json.dumps(stats, indent=2))
    print(stats)


if __name__ == "__main__":
    main()
