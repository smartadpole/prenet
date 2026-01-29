#!/usr/bin/env python3
# encoding: utf-8
import sys
import os
import argparse
import glob
import shutil
import pandas as pd
import fastdup
from utils.file import mkdir_simple
from tools.eval import load_test_file
from utils.logger import logger_manager
from tqdm import tqdm
from utils.utils import timeit
from contextlib import contextmanager

# 图像后缀
IMG_EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')

# todo: smart 2026-01-28 19:34 - unbelievable fix for suppress stdout/stderr of fastdup
@contextmanager
def suppress_stdout_stderr():
    """
    底层文件描述符级别重定向，能够拦截 C/C++ 库的 printf 输出
    """
    # 保存原始的 stdout 和 stderr 文件描述符
    save_stdout_fd = os.dup(sys.stdout.fileno())
    save_stderr_fd = os.dup(sys.stderr.fileno())

    try:
        # 打开 /dev/null
        with open(os.devnull, 'w') as fnull:
            # 将 stdout 和 stderr 强制指向 /dev/null
            os.dup2(fnull.fileno(), sys.stdout.fileno())
            os.dup2(fnull.fileno(), sys.stderr.fileno())
            yield
    finally:
        # 恢复原始的文件描述符
        os.dup2(save_stdout_fd, sys.stdout.fileno())
        os.dup2(save_stderr_fd, sys.stderr.fileno())
        # 关闭备份的描述符
        os.close(save_stdout_fd)
        os.close(save_stderr_fd)

logger_manager.set_log_level(level="debug")


def load_data_from_file(file_path):
    """
    解析文件列表文本：路径, id, class_name
    【修复】：显式返回 pandas.DataFrame
    """
    if not os.path.exists(file_path):
        print(f"文件不存在: {file_path}")
        return None

    print(f"正在加载文件列表: {file_path}")
    file_list = load_test_file(file_path)
    if not file_list:
        return None

    # 构造 DataFrame
    df = pd.DataFrame(file_list, columns=['filename', 'label_id', 'class_name'])
    print(f"✓ 成功加载文件列表: {file_path} (共 {len(df)} 条记录)")
    return df

def _write_filelist(output_path, sampled_data):
    """输出采样后的 list 文件"""
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in sampled_data:
            f.write(f"{item['filename']},{item['label_id']},{item['class_name']}\n")
    print(f"✓ 结果已写入: {output_path} (共 {len(sampled_data)} 条)")

def perform_sampling_from_df(df_clusters, num_templates, edge_ratio, method):
    """从聚类结果 DataFrame 中执行采样"""
    if len(df_clusters) <= num_templates:
        return df_clusters

    if method == "center":
        # 选出每个簇中距离中心最近的
        return df_clusters.loc[df_clusters.groupby('cluster')['distance'].idxmin()].head(num_templates)
    else:
        # 混合采样逻辑
        num_edges = max(1, int(num_templates * edge_ratio))
        num_centers = num_templates - num_edges

        centers = df_clusters.loc[df_clusters.groupby('cluster')['distance'].idxmin()].head(num_centers)

        # 挑选分布最散的簇，取其边缘样本
        cluster_var = df_clusters.groupby('cluster')['distance'].mean().sort_values(ascending=False)
        target_clusters = cluster_var.head(num_edges).index

        edges = []
        for c_id in target_clusters:
            edge_sample = df_clusters[df_clusters['cluster'] == c_id].nlargest(1, 'distance')
            edges.append(edge_sample)

        df_edges = pd.concat(edges) if edges else pd.DataFrame()
        return pd.concat([centers, df_edges]).head(num_templates)

def _normalize_kmeans_df(df_clusters):
    if 'filename' not in df_clusters.columns:
        for alt in ('impath', 'filepath', 'path', 'image', 'img_path'):
            if alt in df_clusters.columns:
                df_clusters = df_clusters.rename(columns={alt: 'filename'})
                break
    if 'distance' not in df_clusters.columns and 'dist' in df_clusters.columns:
        df_clusters = df_clusters.rename(columns={'dist': 'distance'})
    if 'cluster' not in df_clusters.columns and 'cluster_id' in df_clusters.columns:
        df_clusters = df_clusters.rename(columns={'cluster_id': 'cluster'})
    return df_clusters

def _pick_assignments_csv(work_dir):
    default_path = os.path.join(work_dir, "kmeans_assignments.csv")
    if os.path.exists(default_path):
        return default_path
    candidates = glob.glob(os.path.join(work_dir, "*kmeans*assignments*.csv"))
    return candidates[0] if candidates else None

def _run_fastdup_kmeans(class_work_dir, image_paths, num_clusters, num_em_iter):
    mkdir_simple(class_work_dir)
    with suppress_stdout_stderr():
        fastdup.run_kmeans(
            input_dir=image_paths,
            work_dir=class_work_dir,
            num_clusters=num_clusters,
            num_em_iter=num_em_iter,
            verbose=False
        )

    return _pick_assignments_csv(class_work_dir)

# @timeit(1)
def process_by_dataframe(df_list, output_file, method, num_templates, edge_ratio, num_em_iter):
    if df_list is None or df_list.empty:
        return

    df_list = df_list[['filename', 'label_id', 'class_name']].copy()
    df_list['filename'] = df_list['filename'].apply(lambda p: os.path.abspath(str(p)))

    # 预检查文件是否存在
    df_list['exists'] = df_list['filename'].apply(os.path.exists)
    df_valid = df_list[df_list['exists']].copy()

    if df_valid.empty:
        print("错误: 没有有效图片路径", level="error")
        return

    output_dir = os.path.dirname(output_file) or "."
    work_root = os.path.join(output_dir, "work_dirs")
    if os.path.isdir(work_root):
        shutil.rmtree(work_root, ignore_errors=True)
    all_sampled_records = []

    # 按类别进行组内独立聚类，防止特征跨类干扰
    pbar = tqdm(df_valid.groupby(['label_id', 'class_name'], sort=False), desc="[采样进度]", unit="类")

    for (label_id, class_name), group in pbar:
        # 在进度条右侧动态显示类别和数量
        pbar.set_postfix_str(f"当前: {class_name} ({len(group)}张)")

        if len(group) <= num_templates or len(group) < 10:
            all_sampled_records.extend(group[['filename', 'label_id', 'class_name']].to_dict('records'))
            continue

        # 为该类创建独立的隔离工作目录
        class_work_dir = os.path.join(output_dir, "work_dirs", f"label_{label_id}")
        file_list = group['filename'].tolist()

        assignments_path = _run_fastdup_kmeans(
            class_work_dir=class_work_dir,
            image_paths=file_list,
            num_clusters=num_templates,
            num_em_iter=num_em_iter,
        )

        # 读取并采样结果
        if assignments_path and os.path.exists(assignments_path):
            df_c = pd.read_csv(assignments_path)
            df_c = _normalize_kmeans_df(df_c)
            if not {'filename', 'cluster', 'distance'}.issubset(df_c.columns):
                print(f"[Warning] kmeans_assignments 缺少必要列，跳过该类: {class_name}")
                continue
            if 'filename' in df_c.columns and 'distance' in df_c.columns:
                df_c = df_c.loc[df_c.groupby('filename')['distance'].idxmin()]

            sampled_df = perform_sampling_from_df(df_c, num_templates, edge_ratio, method)

            for _, s_row in sampled_df.iterrows():
                all_sampled_records.append({
                    'filename': s_row['filename'],
                    'label_id': label_id,
                    'class_name': class_name
                })

    try:
        # 输出结果 txt
        with open(output_file, 'w', encoding='utf-8') as f:
            for item in all_sampled_records:
                f.write(f"{item['filename']},{item['label_id']},{item['class_name']}\n")
        print(f"\n✓ 任务完成！采样结果已保存至: {output_file}", level="info")
    finally:
        # 清理临时缓存目录
        if os.path.isdir(work_root):
            shutil.rmtree(work_root, ignore_errors=True)
    print(f"  - 输入类别数: {df_valid['label_id'].nunique()}")
    print(f"  - 采样类别数: {len(set([r['label_id'] for r in all_sampled_records]))}")
    print(f"  - 总采样数量: {len(all_sampled_records)} 条记录")

def process_by_file(list_file, output_file, method, num_templates, edge_ratio, num_em_iter):
    """
    基于列表文件的采样逻辑：按类独立聚类
    """
    df_list = load_data_from_file(list_file)
    if df_list is None:
        return
    process_by_dataframe(df_list, output_file, method, num_templates, edge_ratio, num_em_iter)

def process_by_directory(root_dir, output_file, method, num_templates, edge_ratio, num_em_iter):
    """递归扫描目录，自动构造 list 并调用 process_by_file"""
    dir_to_images = {}
    print(f"正在扫描目录: {root_dir}")
    for root, dirs, files in os.walk(root_dir):
        # 判定叶子节点：包含图片且图片数较多
        images = [f for f in files if f.lower().endswith(IMG_EXTS)]
        if images:
            dir_to_images[root] = images

    if not dir_to_images:
        print("未发现有效图片")
        return

    # 仅保留叶子目录：其子目录不再包含图片
    image_dirs = list(dir_to_images.keys())
    leaf_dirs = []
    for d in image_dirs:
        has_child = any(other != d and other.startswith(d + os.sep) for other in image_dirs)
        if not has_child:
            leaf_dirs.append(d)

    leaf_dirs = sorted(leaf_dirs)
    label_map = {d: idx for idx, d in enumerate(leaf_dirs)}

    data = []
    for d in leaf_dirs:
        class_name = os.path.basename(d)
        label_id = label_map[d]
        for img in dir_to_images[d]:
            data.append([os.path.abspath(os.path.join(d, img)), label_id, class_name])

    if not data:
        print("未发现有效图片")
        return

    df_list = pd.DataFrame(data, columns=['filename', 'label_id', 'class_name'])
    process_by_dataframe(df_list, output_file, method, num_templates, edge_ratio, num_em_iter)

def main():
    parser = argparse.ArgumentParser(description="智慧门店模板库采样工具 (已修复版)")
    parser.add_argument("--input", type=str, required=True, help="输入路径 (txt列表或根目录)")
    parser.add_argument("--output", type=str, default=None, help="输出 txt 路径")
    parser.add_argument("--method", type=str, choices=["center", "hybrid"], default="center")
    parser.add_argument("--num_templates", type=int, default=20)
    parser.add_argument("--edge_ratio", type=float, default=0.2)
    parser.add_argument("--num_em_iter", type=int, default=30)

    args = parser.parse_args()

    if args.output:
        output_file = args.output
    else:
        input_path = os.path.normpath(args.input)

        if os.path.isfile(input_path):
            file_root, file_ext = os.path.splitext(input_path)
            output_file = f"{file_root}_templates{file_ext}"
        else:
            output_file = os.path.join(input_path, "templates.txt")

    output_file = os.path.abspath(output_file)
    print(f"计划输出路径: {output_file}")

    if os.path.isfile(args.input):
        process_by_file(args.input, output_file, args.method, args.num_templates, args.edge_ratio, args.num_em_iter)
    else:
        process_by_directory(args.input, output_file, args.method, args.num_templates, args.edge_ratio, args.num_em_iter)

if __name__ == '__main__':
    main()
