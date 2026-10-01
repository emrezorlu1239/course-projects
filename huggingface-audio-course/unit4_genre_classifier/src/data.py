"""GTZAN loading, clip-level split and AST feature caching.

datasets>=4 dropped loading scripts and decodes audio through torchcodec. To stay
independent of both, we read the Hub's auto-converted parquet branch, keep the raw
audio bytes (decode=False) and decode them with soundfile.
"""

import io
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
from datasets import Audio, load_dataset
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from transformers import ASTFeatureExtractor

DATASET_ID = "marsyas/gtzan"
TARGET_SR = 16_000
CHUNK_SECONDS = 10.0


def load_gtzan():
    ds = load_dataset(DATASET_ID, revision="refs/convert/parquet", split="train")
    return ds.cast_column("audio", Audio(decode=False))


def decode(audio: dict) -> np.ndarray:
    """Raw bytes -> mono float32 waveform at 16 kHz."""
    y, sr = sf.read(io.BytesIO(audio["bytes"]), dtype="float32", always_2d=True)
    y = y.mean(axis=1)
    if sr != TARGET_SR:
        y = librosa.resample(y, orig_sr=sr, target_sr=TARGET_SR)
    return y


def chunk(y: np.ndarray) -> list[np.ndarray]:
    """Split a ~30 s clip into non-overlapping 10 s chunks (last partial chunk kept if >= 5 s)."""
    n = int(CHUNK_SECONDS * TARGET_SR)
    chunks = [y[i : i + n] for i in range(0, len(y), n)]
    return [c for c in chunks if len(c) >= n // 2]


def split_clips(labels: list[int], val_size: float, seed: int):
    """Stratified split over CLIP indices, so chunks of one clip never cross train/val."""
    idx = np.arange(len(labels))
    return train_test_split(idx, test_size=val_size, stratify=labels, random_state=seed)


def build_cache(cache_dir: Path, model_id: str, val_size: float, seed: int) -> dict:
    """Compute AST log-mel fbank features for every chunk once and store them as float16."""
    cache = cache_dir / f"gtzan_ast_val{val_size}_seed{seed}.npz"
    if cache.exists():
        return dict(np.load(cache, allow_pickle=True))

    ds = load_gtzan()
    fe = ASTFeatureExtractor.from_pretrained(model_id)
    labels = ds["genre"]
    train_idx, val_idx = split_clips(labels, val_size, seed)
    split_of = {int(i): "train" for i in train_idx} | {int(i): "val" for i in val_idx}

    feats, ys, clip_ids, splits = [], [], [], []
    for i in tqdm(range(len(ds)), desc="features"):
        ex = ds[i]
        try:
            y = decode(ex["audio"])
        except Exception as e:  # GTZAN ships one known-corrupt file; skip anything undecodable
            print(f"skip clip {i} ({ex['file']}): {e}")
            continue
        for c in chunk(y):
            x = fe(c, sampling_rate=TARGET_SR, return_tensors="np")["input_values"][0]
            feats.append(x.astype(np.float16))
            ys.append(ex["genre"])
            clip_ids.append(i)
            splits.append(split_of[i])

    out = {
        "features": np.stack(feats),
        "labels": np.array(ys, dtype=np.int64),
        "clip_ids": np.array(clip_ids, dtype=np.int64),
        "splits": np.array(splits),
        "label_names": np.array(ds.features["genre"].names),
    }
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.savez(cache, **out)
    return out
