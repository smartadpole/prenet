#!/bin/bash

# --- 1. 定义基础路径和变量 ---
BASE_DIR="/home/user/data/video/2025-12-24-all-stall-result3"
PYTHON_BIN="/home/user/sunhao/lib/torch240cu128/bin/python3"
PY_SCRIPT="/home/user/sunhao/CODE/prenet/tools/test_full.py"
MODEL_PATH="/home/user/data/Model/Classify/D1.1.7/best_val_dinov2_arcface_small.pt"
LABEL_PATH="/home/user/data/image/classify/files/train/v1.0.2/label.txt"
SAVE_DIR="/home/user/sunhao/Result/"

# --- 2. 遍历目录并执行 ---
# 遍历 BASE_DIR 下的所有子目录
for dir_path in "$BASE_DIR"/*/; do
    # 去掉路径末尾的斜杠，提取纯目录名（例如：自选-刺身小吃鲜切工坊-新）
    dir_name=$(basename "$dir_path")

    # 根据你的指令模板，CSV 文件位于该目录下，且文件名包含目录名后缀
    # 格式：/目录路径/目录名_count_results.csv
    CSV_FILE="${dir_path}${dir_name}_count_results.csv"

    # 检查 CSV 文件是否存在，存在则执行
    if [ -f "$CSV_FILE" ]; then
        echo "========================================================="
        echo "正在处理目录: $dir_name"
        echo "输入文件: $CSV_FILE"

        $PYTHON_BIN $PY_SCRIPT \
            --input_csv "$CSV_FILE" \
            --suffix label_hao \
            --model_path "$MODEL_PATH" \
            --batch_size 32 \
            --label "$LABEL_PATH" \
            --temp_save_dir "$SAVE_DIR"

        echo "目录 $dir_name 处理完成。"
    else
        echo "警告: 跳过 $dir_name，未找到文件 $CSV_FILE"
    fi
done

echo "所有任务执行完毕。"