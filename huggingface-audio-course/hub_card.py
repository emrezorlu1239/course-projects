"""Shared helpers to publish a finished Trainer run to the Hub with a complete model card."""

import re
from pathlib import Path

import torch
from huggingface_hub import HfApi
from transformers import TrainerState

PLACEHOLDER = "More information needed"
AUTOGEN_COMMENT = re.compile(r"<!-- This model card has been generated automatically.*?-->\n*", re.S)


def restore_trainer(trainer_cls, model, processing_class, out_dir: Path, repo_id: str):
    """Rebuild a Trainer with the original TrainingArguments and training log of a finished run."""
    args = torch.load(out_dir / "training_args.bin", weights_only=False)
    args.output_dir = str(out_dir)
    args.hub_model_id = repo_id
    args.push_to_hub = False
    args.eval_strategy = "no"  # no eval set is attached when only publishing
    trainer = trainer_cls(model=model, args=args, processing_class=processing_class)
    trainer.state = TrainerState.load_from_json(str(out_dir / "trainer_state.json"))
    return trainer


def write_card(trainer, card_kwargs: dict, sections: dict[str, str], extra_tags: list[str] | None = None) -> Path:
    """Generate the Trainer model card, then fill the 'More information needed' sections.

    The auto-generated metric lines ("- Accuracy: 0.xx" / "- Wer: 0.xx") stay first in the card,
    which is what the course's progress checker parses.
    """
    kwargs = dict(card_kwargs)
    if extra_tags:
        kwargs["tags"] = extra_tags
    trainer.create_model_card(**kwargs)
    path = Path(trainer.args.output_dir) / "README.md"
    text = AUTOGEN_COMMENT.sub("", path.read_text(encoding="utf-8"))
    for heading, body in sections.items():
        text = text.replace(f"## {heading}\n\n{PLACEHOLDER}", f"## {heading}\n\n{body.strip()}")
    path.write_text(text, encoding="utf-8")
    return path


def list_upload(out_dir: Path) -> list[tuple[str, float]]:
    files = []
    for f in sorted(out_dir.rglob("*")):
        rel = f.relative_to(out_dir)
        if f.is_file() and not rel.parts[0].startswith(("checkpoint-", "_")):
            files.append((rel.as_posix(), f.stat().st_size / 1e6))
    return files


def upload(trainer, commit_message: str):
    """Same upload as Trainer.push_to_hub (minus regenerating the card): save model, upload output_dir."""
    trainer.save_model()
    api = HfApi()
    repo_id = trainer.args.hub_model_id
    api.create_repo(repo_id, exist_ok=True, private=False)
    return api.upload_folder(
        repo_id=repo_id,
        folder_path=trainer.args.output_dir,
        commit_message=commit_message,
        ignore_patterns=["_*", "checkpoint-*", "checkpoint-*/**"],
    )
