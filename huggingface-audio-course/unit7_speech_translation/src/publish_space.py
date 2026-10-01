"""Duplicate the course template Space and upload our Turkish STST demo into it (public, free CPU).

Usage:
    python unit7_speech_translation/src/publish_space.py --dry-run
    python unit7_speech_translation/src/publish_space.py
"""

import argparse
from pathlib import Path

from huggingface_hub import HfApi, whoami

TEMPLATE = "course-demos/speech-to-speech-translation"
SPACE_NAME = "speech-to-speech-translation-tr"
SPACE_DIR = Path(__file__).resolve().parents[1] / "space"
FILES = ["app.py", "requirements.txt", "README.md", "packages.txt", "example.wav"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    repo_id = f"{whoami()['name']}/{SPACE_NAME}"
    print(f"Space: https://huggingface.co/spaces/{repo_id} (public, cpu-basic), duplicated from {TEMPLATE}")
    for f in FILES:
        print(f"  {f}  {(SPACE_DIR / f).stat().st_size / 1e3:.1f} kB")
    if args.dry_run:
        return

    api = HfApi()
    if not api.repo_exists(repo_id, repo_type="space"):
        # duplicate_space needs a PRO plan for gradio hardware; a plain Space repo + the template files is equivalent.
        api.create_repo(repo_id, repo_type="space", space_sdk="gradio", private=False, exist_ok=True)
    info = api.upload_folder(
        repo_id=repo_id,
        repo_type="space",
        folder_path=str(SPACE_DIR),
        allow_patterns=FILES,
        commit_message="Translate any language to Turkish speech (Whisper -> Marian -> MMS-TTS)",
    )
    print(info)


if __name__ == "__main__":
    main()
