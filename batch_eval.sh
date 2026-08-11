#!/usr/bin/env bash

set -e

SAMPLE_MODE="random"
TIMES=1
LOAD_MODEL_PATH=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --load-model-path)
            LOAD_MODEL_PATH="$2"
            shift 2
            ;;
        --times)
            TIMES="$2"
            shift 2
            ;;
        --sample-mode)
            SAMPLE_MODE="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

if [[ -z "$LOAD_MODEL_PATH" ]]; then
    echo "Error: --load-model-path is required"
    exit 1
fi

source .venv/bin/activate

for ((i=1; i<=TIMES; i++)); do
    echo "========================================"
    echo "Run $i / $TIMES"
    echo "========================================"

    puffer eval adv_drive \
        --eval.sample-mode "$SAMPLE_MODE" \
        --eval.map-dir "resources/drive/binaries/selfplay_train" \
        --load-model-path "$LOAD_MODEL_PATH"
done