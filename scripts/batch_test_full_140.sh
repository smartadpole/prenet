#!/bin/bash

# Default paths and options
PYTHON_BIN="/home/user/sunhao/lib/torch240cu128/bin/python3"
PY_SCRIPT="/home/user/sunhao/CODE/prenet/tools/test_full.py"
MODEL_PATH="/home/user/data/Model/Classify/D1.1.11/best_val_dinov2_arcface_small.pt"
LABEL_PATH="/home/user/data/image/classify/files/train/v1.0.5/label.txt"
RUN_TS=$(date +"%Y%m%d_%H%M")
SAVE_ROOT="/home/user/sunhao/Result/${RUN_TS}"
SUFFIX="label_hao"
BATCH_SIZE=32
CROP_DIR=""

# Open-set options
OPEN_SET=0
TEMPLATE_PATH=""
ALLOW_UNKNOWN=0

usage() {
    cat <<EOF
Usage: $0 [options]

Options:
  --model_path <path>       Model checkpoint path
  --label_file <path>       Label file path (optional)
  --suffix <name>           Output CSV suffix (default: ${SUFFIX})
  --batch_size <n>          Batch size (default: ${BATCH_SIZE})
  --save_root <dir>         Base directory for saving cropped images (optional)
  --crop_dir <dir>          Temporary directory for saving cropped images (optional)
  --template_path <path>    Template file or directory for open-set retrieval
  --allow_unknown           Allow Unknown/Rejected output in open-set mode
  -h, --help                Show this help
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model_path) MODEL_PATH="$2"; shift 2 ;;
        --label_file) LABEL_PATH="$2"; shift 2 ;;
        --suffix) SUFFIX="$2"; shift 2 ;;
        --batch_size) BATCH_SIZE="$2"; shift 2 ;;
        --save_root) SAVE_ROOT="$2"; shift 2 ;;
        --crop_dir) CROP_DIR="$2"; shift 2 ;;
        --template_path) TEMPLATE_PATH="$2"; shift 2 ;;
        --allow_unknown) ALLOW_UNKNOWN=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown option: $1"; usage; exit 1 ;;
    esac
done

if [[ -n "$TEMPLATE_PATH" ]]; then
    OPEN_SET=1
fi

if [[ $OPEN_SET -eq 1 && -z "$TEMPLATE_PATH" ]]; then
    echo "template_path is required when open_set is enabled"
    exit 1
fi

find . -type f -name "*count_results.csv" | while read -r CSV_FILE; do
    dir_path=$(dirname "$CSV_FILE")
    rel_dir="${dir_path#./}"
    save_dir="${SAVE_ROOT}/"

    echo "========================================================="
    echo "Processing: ${CSV_FILE}"
    echo "Save dir: ${save_dir}"

    CMD=(
        "$PYTHON_BIN" "$PY_SCRIPT"
        --input_csv "$CSV_FILE"
        --suffix "$SUFFIX"
        --model_path "$MODEL_PATH"
        --batch_size "$BATCH_SIZE"
    )

    if [[ -n "$LABEL_PATH" ]]; then
        CMD+=(--label_file "$LABEL_PATH")
    fi

    if [[ $OPEN_SET -eq 1 ]]; then
        CMD+=(--template_path "$TEMPLATE_PATH")
    fi

    if [[ $ALLOW_UNKNOWN -eq 1 ]]; then
        CMD+=(--allow_unknown)
    fi

    if [[ -n "$CROP_DIR" ]]; then
        CMD+=(--crop_dir "$CROP_DIR")
    fi

    if [[ -n "$SAVE_ROOT" ]]; then
        CMD+=(--save_dir "$save_dir")
    fi

    "${CMD[@]}"

    echo "Done: ${CSV_FILE}"
done

echo "All tasks completed."
