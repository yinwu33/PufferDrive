#!/usr/bin/env bash
set -euo pipefail

usage() {
    echo "Usage: $0 <env_name> <bin_path> [puffer eval args...]"
    echo
    echo "Example:"
    echo "  $0 cond_drive resources/drive/binaries/training/map_123.bin --load-model-path ./experiments/cond_drive_RUN_ID.pt"
}

if [[ $# -lt 2 ]]; then
    usage >&2
    exit 2
fi

env_name="$1"
bin_path="$2"
shift 2

if [[ ! -f "$bin_path" ]]; then
    echo "Error: bin_path does not exist or is not a file: $bin_path" >&2
    exit 1
fi

if command -v uuidgen >/dev/null 2>&1; then
    tmp_id="$(uuidgen)"
else
    tmp_id="$(python3 -c 'import uuid; print(uuid.uuid4())')"
fi

tmp_dir="/tmp/${tmp_id}"
mkdir -p "$tmp_dir"
cp -- "$bin_path" "$tmp_dir/map_000.bin"

echo "Using temporary map directory: $tmp_dir"
echo "Copied map to: $tmp_dir/map_000.bin"

exec puffer eval "$env_name" \
    --eval.map-dir "$tmp_dir" \
    --eval.sample-mode random \
    --env.num-maps 1 \
    "$@"
