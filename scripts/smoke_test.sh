#!/usr/bin/env bash
# Offline smoke tests: no API key, external network, or paid requests required.
set -euo pipefail
cd "$(dirname "$0")/.."
python -m unittest discover -s tests -v
