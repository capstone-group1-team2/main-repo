"""Downloads the raw Bitext Customer Support LLM Chatbot Training Dataset
from Hugging Face and writes it to `data/bitext_customer_support.csv`

Dataset: bitext/Bitext-customer-support-llm-chatbot-training-dataset
License: Community Data License Agreement – Sharing, version 1.0 (CDLA-Sharing-1.0)
https://cdla.dev/sharing-1-0/
Attribution: Bitext Innovations International, S.L.

"""

from __future__ import annotations

import argparse
import shutil

from huggingface_hub import hf_hub_download

_REPO_ID = "bitext/Bitext-customer-support-llm-chatbot-training-dataset"
_REPO_FILENAME = "Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11.csv"
_DEFAULT_OUT = "data/bitext_customer_support.csv"


def fetch(out_path: str = _DEFAULT_OUT) -> str:
    downloaded_path = hf_hub_download(repo_id=_REPO_ID, filename=_REPO_FILENAME, repo_type="dataset")
    shutil.copyfile(downloaded_path, out_path)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        default=_DEFAULT_OUT,
        help=f"Output CSV path (default: {_DEFAULT_OUT}). Use a temp path to verify "
        "reproducibility without overwriting the real file.",
    )
    args = parser.parse_args()
    out_path = fetch(args.out)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
