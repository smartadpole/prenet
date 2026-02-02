#!/bin/bash

OUTPUT_NAME="output.txt"
TRAIN_NAME="${OUTPUT_NAME%.txt}_train.txt"
TEST_NAME="${OUTPUT_NAME%.txt}_test.txt"

# 1. 定义统计函数 (替代原本的 alias)
function countfile() {
    # $1 是传入的文件名
    sed -r 's|/[^/,]+,|,|' "$1"
}

function count() {
    # 接收管道输入进行统计
    sort | uniq -c | sort -nr
}

# ============================================================
# 第一部分：内置核心函数 (getlabeleddir)
# ============================================================
# 清理别名，防止冲突
unalias getlabeleddir 2>/dev/null
function getlabeleddir {
    local processed_json="${1:-.processed_folders.json}"
    local parent_dir

    [[ -f "$processed_json" ]] || return 1
    parent_dir="$(dirname "$processed_json")"

    # 直接读取 JSON 列表
    jq -r '.folders[]' "$processed_json" | while read -r folder; do
        # 排除包含 temp 或 trash 的标签名
        [[ "$folder" =~ temp|trash ]] && continue

        # 精确拼接路径并检查该目录是否存在
        local target_path="$parent_dir/$folder"
        if [[ -d "$target_path" ]]; then
            echo "$target_path/"
        fi
    done
}

function split_data {
    if [ "$#" -lt 2 ]; then
        echo "用法: split_data <文件名> <第一份文件的比例 0-1>"
        return 1
    fi
    
    local file=$1
    local ratio=$2
    local out1=${TRAIN_NAME}
    local out2=${TEST_NAME}
    
    echo "正在处理: $file (比例: $ratio)"
    shuf "$file" | awk -v r="$ratio" -v t=$(wc -l < "$file") -v f1="$out1" -v f2="$out2" 'NR <= t*r {print > f1; next} {print > f2}'
    
    echo "完成! 生成了: $out1 和 $out2"
    wc -l "$out1" "$out2"
}

# ============================================================
# 第二部分：主逻辑
# ============================================================

SEARCH_ROOT="${1:-.}"

echo "[INFO] 正在扫描根目录: $SEARCH_ROOT" >&2

# 1. 查找 JSON -> 2. 提取目录 -> 3. 找图片 -> 4. 排序 -> 5. 逻辑处理
find "$SEARCH_ROOT" -name ".processed_folders.json" | \
while read -r json_file; do
    echo "[Processing] $json_file" >&2
    getlabeleddir "$json_file"
done | \
while read -r valid_dir; do
    if [[ -d "$valid_dir" ]]; then
        find "$valid_dir" -maxdepth 1 -type f \( -iname "*.jpg" -o -iname "*.png" -o -iname "*.jpeg" \)
    fi
done | \
sort | \
awk -F/ '
BEGIN {
    OFS=","; 
    id_counter=0;    # 全局 ID 计数器
    unk_counter=0;   # 待定编号计数器
}

{
    # $0 是文件绝对路径
    # $(NF-1) 是父文件夹的名字
    curr_dir = $(NF-1)

    # 获取文件夹的完整绝对路径 (用于区分不同的纯数字文件夹)
    dir_full_path = $0
    sub(/\/[^\/]+$/, "", dir_full_path)

    # --- 1. 提取标签名字 (label_name) ---
    
    if (curr_dir ~ /_/) {
        # 逻辑：有下划线，取最后一段
        n = split(curr_dir, parts, "_")
        label_name = parts[n]
    } 
    else if (curr_dir ~ /^[0-9]+$/) {
        # 逻辑：纯数字，使用待定递增编号
        # 使用文件夹完整路径作为 Key，确保同一个文件夹里的图片都叫同一个 "待定N"
        if (!(dir_full_path in path2unk)) {
            unk_counter++
            path2unk[dir_full_path] = "待定" unk_counter
        }
        label_name = path2unk[dir_full_path]
    } 
    else {
        # 逻辑：普通字符串，直接使用
        label_name = curr_dir
    }

    # --- 2. 分配 ID ---
    
    # 只要标签名字相同，就分配同一个 ID
    if (!(label_name in name2id)) {
        id_counter++
        name2id[label_name] = id_counter
    }

    # --- 3. 输出 ---
    # 格式: 绝对路径, ID, 标签名字
    print $0, name2id[label_name], label_name

}' > ${OUTPUT_NAME}

echo "----------------------------------------"
echo "全部完成！"
echo "输出文件: ${OUTPUT_NAME}"
echo "总图片数: $(wc -l < ${OUTPUT_NAME})"
echo "总类别数: $(awk -F, '{print $3}' ${OUTPUT_NAME} | sort -u | wc -l)"

# 拆分训练集
split_data ${OUTPUT_NAME}  0.7

# 统计数量
countfile ${OUTPUT_NAME} | count > count_file.txt 
countfile ${TRAIN_NAME} | count > count_train.txt 
countfile ${TEST_NAME} | count > count_val.txt 
countfile ${OUTPUT_NAME} | count | cut -d , -f 2- | sort | uniq | sort -n  > label.txt
countfile ${OUTPUT_NAME}  | rev | cut -d / -f 1 | rev | count > count.txt

ln -sf ${TRAIN_NAME} train.txt
ln -sf ${TEST_NAME} val.txt

