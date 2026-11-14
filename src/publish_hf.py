"""Uploads the trained forest and the per-second table to Hugging Face as two PRIVATE repositories.

Before running: create a free account on huggingface.co, run `hf auth login` once (it asks for an access
token with write rights from huggingface.co/settings/tokens), and run src/18_train_final.py so the model exists.
Run from the repo root:  python src/publish_hf.py YOUR_HF_USERNAME
Check both pages on the website, then switch them to public there if they look right."""
import sys
from pathlib import Path

from huggingface_hub import HfApi

import config

if len(sys.argv) != 2:
    sys.exit("usage: python src/publish_hf.py YOUR_HF_USERNAME")
user = sys.argv[1]
HF = config.PROJECT / "hf"
files = {"model": [config.PROJECT / "models" / "forest.joblib"],
         "dataset": [config.DATA / "table.csv", config.DATA / "split.csv"]}
missing = [str(p) for ps in files.values() for p in ps if not p.exists()]
if missing:
    sys.exit(f"missing files: {missing}")

api = HfApi()
print("logged in as:", api.whoami()["name"])
repos = {"model": f"{user}/alfa-fault-forest", "dataset": f"{user}/alfa-per-second-table"}
cards = {"model": HF / "MODEL_CARD.md", "dataset": HF / "DATASET_CARD.md"}
for kind, repo in repos.items():
    api.create_repo(repo, repo_type=kind, private=True, exist_ok=True)
    card = cards[kind].read_text(encoding="utf-8").replace("{USER}", user)
    api.upload_file(path_or_fileobj=card.encode("utf-8"), path_in_repo="README.md", repo_id=repo, repo_type=kind)
    for p in files[kind]:
        api.upload_file(path_or_fileobj=str(p), path_in_repo=p.name, repo_id=repo, repo_type=kind)
    prefix = "datasets/" if kind == "dataset" else ""
    print(f"uploaded (private): https://huggingface.co/{prefix}{repo}")
