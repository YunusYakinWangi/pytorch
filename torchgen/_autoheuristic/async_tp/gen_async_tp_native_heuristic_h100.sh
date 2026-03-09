#!/bin/bash
# Generate the AsyncTPNativeH100 heuristic artifact from existing training data.
# Run from: torchgen/_autoheuristic/async_tp/
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

python train_decision_async_tp_native.py \
    async_tp_native_h100_data.txt \
    --heuristic-name AsyncTPNativeH100

echo "Generated artifact: torch/_inductor/autoheuristic/artifacts/_AsyncTPNativeH100.py"
