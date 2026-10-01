"""Reproduce the Unit 7 assessor locally: call the demo's /predict endpoint and run language ID on the output.

The official assessor (huggingface-course/audio-course-u7-assessment) sends `test_short.wav` to
`/predict` and requires facebook/mms-lid-126 to predict a non-English language with score >= 0.5.

Usage:
    python unit7_speech_translation/src/check_space.py --space http://127.0.0.1:7860/ --audio test_short.wav
    python unit7_speech_translation/src/check_space.py --space Zorlu5454/speech-to-speech-translation-tr --audio test_short.wav
"""

import argparse
import json
import shutil
import time
from pathlib import Path

import soundfile as sf
from gradio_client import Client, handle_file
from transformers import pipeline

ROOT = Path(__file__).resolve().parents[2]
THRESHOLD = 0.5


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--space", required=True)
    p.add_argument("--audio", required=True)
    p.add_argument("--tag", default="local")
    args = p.parse_args()

    client = Client(args.space)
    t0 = time.time()
    out_file = client.predict(handle_file(args.audio), api_name="/predict")
    latency = time.time() - t0

    audio, sr = sf.read(out_file)
    lid = pipeline("audio-classification", model="facebook/mms-lid-126", device=-1)
    preds = lid({"array": audio, "sampling_rate": sr})
    top = preds[0]
    passed = top["score"] >= THRESHOLD and top["label"] != "eng"

    results_dir = ROOT / "results"
    shutil.copy(out_file, results_dir / f"unit7_output_{args.tag}.wav")
    result = {
        "space": args.space,
        "input": Path(args.audio).name,
        "latency_s": round(latency, 2),
        "output_seconds": round(len(audio) / sr, 2),
        "sampling_rate": sr,
        "language_id_top5": preds[:5],
        "passes_assessor_check": passed,
    }
    (results_dir / f"unit7_check_{args.tag}.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
