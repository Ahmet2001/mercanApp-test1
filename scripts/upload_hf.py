#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
import tempfile
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser(description="Publish a Mercan .mercan artifact to Hugging Face")
    p.add_argument("model", type=Path, help="path to model.mercan")
    p.add_argument("--repo", help="Hugging Face repo id, e.g. owner/Mercan-0.8B-SFT")
    p.add_argument("--readme", type=Path, default=Path(__file__).resolve().parents[1] / "huggingface" / "README.md")
    p.add_argument("--private", action="store_true")
    args = p.parse_args()

    if not args.model.is_file():
        raise SystemExit(f"model does not exist: {args.model}")
    if args.model.name != "model.mercan":
        print(f"warning: CLI default lookup expects model.mercan, got {args.model.name}")

    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise SystemExit("huggingface_hub is required: python -m pip install -U huggingface_hub") from exc

    api = HfApi()
    who = api.whoami()
    username = who.get("name") or who.get("fullname")
    if not username:
        raise SystemExit("could not determine Hugging Face username from active token")

    repo_id = args.repo or f"{username}/Mercan-0.8B-SFT"
    if "/" not in repo_id:
        repo_id = f"{username}/{repo_id}"

    api.create_repo(repo_id=repo_id, repo_type="model", private=args.private, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="mercan-hf-") as td:
        stage = Path(td)
        shutil.copy2(args.model, stage / "model.mercan")
        sha = args.model.parent / "model.mercan.sha256"
        if sha.is_file():
            shutil.copy2(sha, stage / sha.name)
        manifest = args.model.with_suffix(args.model.suffix + ".json")
        if manifest.is_file():
            shutil.copy2(manifest, stage / manifest.name)
        if args.readme.is_file():
            shutil.copy2(args.readme, stage / "README.md")

        api.upload_folder(
            repo_id=repo_id,
            repo_type="model",
            folder_path=str(stage),
            commit_message="Publish Mercan 0.8B SFT Q4 model",
        )

    print(f"HF_UPLOAD_OK https://huggingface.co/{repo_id}")


if __name__ == "__main__":
    main()
