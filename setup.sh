#!/usr/bin/env bash

set -euo pipefail

py -3.11 -m venv .venv

source .venv/Scripts/activate
pip install -r requirements.txt --break-system-packages
echo "Setup complete."