#!/usr/bin/env python3
# encoding: utf-8
'''
@author: 孙昊
@contact: smartadpole@163.com
@file: gen_sku.py
@time: 2026/01/26 18:00
@desc: Generate template library using fastdup clustering-based sampling for open-set recognition
       Supports center sampling and hybrid sampling (center + edge) strategies
'''
import sys
import os

CURRENT_DIR = os.path.dirname(__file__)
sys.path.append(os.path.join(CURRENT_DIR, '../'))

import argparse
import pandas as pd
import shutil
import fastdup
from utils.file import mkdir_simple


def method2_template_sampling(input_dir, work_dir="work_dir_template", output_dir="template_library", 
                              num_templates=15, num_em_iter=30):
    """
    Improved method 2: Cluster center sampling
    Logic: Cluster images into K clusters, extract 1 image closest to center from each cluster as template
    
    Args:
        input_dir: Input image directory (single category folder)
        work_dir: Fastdup working directory
        output_dir: Output directory for template library
        num_templates: Number of templates to generate (K clusters)
        num_em_iter: Number of EM iterations for KMeans (default: 30 for complex environments)
    """
    print("=" * 60)
    print(f"Generating template library for directory: {input_dir}")
    print("=" * 60)
    
    mkdir_simple(work_dir)
    mkdir_simple(output_dir)

    # Check if input directory exists and has images
    if not os.path.exists(input_dir):
        print(f"Error: Input directory does not exist: {input_dir}")
        return False
    
    # Count images in input directory
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
    image_files = [f for f in os.listdir(input_dir) 
                   if f.lower().endswith(image_extensions)]
    
    if len(image_files) == 0:
        print(f"Warning: No images found in {input_dir}")
        return False
    
    if len(image_files) < num_templates:
        print(f"Warning: Only {len(image_files)} images found, less than required {num_templates} templates")
        print(f"Using all {len(image_files)} images as templates")
        # Copy all images directly
        category_name = os.path.basename(input_dir.rstrip('/\\'))
        target_dir = os.path.join(output_dir, category_name)
        os.makedirs(target_dir, exist_ok=True)
        
        for img_file in image_files:
            src_path = os.path.join(input_dir, img_file)
            shutil.copy2(src_path, os.path.join(target_dir, img_file))
        
        print(f"✓ {category_name} template library generated at: {target_dir}")
        return True

    # 1. Run KMeans clustering
    # num_clusters is the number of templates we want
    print(f"\nRunning KMeans clustering (clusters: {num_templates}, iterations: {num_em_iter})...")
    ret = fastdup.run_kmeans(
        input_dir=input_dir,
        work_dir=work_dir,
        num_clusters=num_templates,
        num_em_iter=num_em_iter,  # Increase iterations for light and smoke interference
        verbose=False
    )

    if ret != 0:
        print("Error: Clustering failed")
        return False

    # 2. Read clustering results
    kmeans_assignments_file = os.path.join(work_dir, "kmeans_assignments.csv")
    if not os.path.exists(kmeans_assignments_file):
        print(f"Error: Clustering result file not found: {kmeans_assignments_file}")
        return False

    print("Reading clustering results...")
    df_clusters = pd.read_csv(kmeans_assignments_file)
    
    # Ensure each image belongs to only one cluster (keep the one with minimum distance)
    original_count = len(df_clusters)
    df_clusters = df_clusters.loc[df_clusters.groupby('filename')['distance'].idxmin()]
    
    if original_count != len(df_clusters):
        print(f"Removed duplicates: {len(df_clusters)} unique images (from {original_count} records)")

    # 3. Core sampling logic: Find the sample with minimum distance in each cluster
    # Group by cluster and find the row with minimum distance
    df_templates = df_clusters.loc[df_clusters.groupby('cluster')['distance'].idxmin()]

    print(f"Clustering completed: Selected {len(df_templates)} representative templates from {len(df_clusters)} images")

    # 4. Copy sampled template images
    category_name = os.path.basename(input_dir.rstrip('/\\'))
    target_dir = os.path.join(output_dir, category_name)
    os.makedirs(target_dir, exist_ok=True)

    copied_count = 0
    for idx, row in df_templates.iterrows():
        src_path = row['filename']
        if not os.path.exists(src_path):
            print(f"Warning: File not found, skipping: {src_path}")
            continue
        
        # Rename to identify which cluster it belongs to
        file_ext = os.path.splitext(src_path)[1]
        dst_name = f"template_c{row['cluster']}{file_ext}"
        dst_path = os.path.join(target_dir, dst_name)
        shutil.copy2(src_path, dst_path)
        copied_count += 1
        
    print(f"✓ {category_name} template library generated at: {target_dir} ({copied_count} templates)")
    return True


def method2_hybrid_sampling(input_dir, work_dir="work_dir_hybrid", output_dir="hybrid_templates", 
                           total_count=15, edge_ratio=0.2, num_em_iter=30):
    """
    Hybrid sampling scheme: Center samples + Edge samples
    Args:
        input_dir: Input image directory (single category folder)
        work_dir: Fastdup working directory
        output_dir: Output directory for hybrid templates
        total_count: Total number of templates needed
        edge_ratio: Ratio of edge samples, 0.2 means 20% of images are extreme cases
        num_em_iter: Number of EM iterations for KMeans
    """
    import numpy as np
    
    mkdir_simple(work_dir)
    mkdir_simple(output_dir)

    # Check if input directory exists
    if not os.path.exists(input_dir):
        print(f"Error: Input directory does not exist: {input_dir}")
        return False
    
    # Count images
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
    image_files = [f for f in os.listdir(input_dir) 
                   if f.lower().endswith(image_extensions)]
    
    if len(image_files) == 0:
        print(f"Warning: No images found in {input_dir}")
        return False
    
    if len(image_files) < total_count:
        print(f"Warning: Only {len(image_files)} images found, less than required {total_count} templates")
        print(f"Using all {len(image_files)} images as templates")
        category_name = os.path.basename(input_dir.rstrip('/\\'))
        target_dir = os.path.join(output_dir, category_name)
        os.makedirs(target_dir, exist_ok=True)
        
        for img_file in image_files:
            src_path = os.path.join(input_dir, img_file)
            shutil.copy2(src_path, os.path.join(target_dir, img_file))
        
        print(f"✓ {category_name} template library generated at: {target_dir}")
        return True

    # 1. Increase cluster count to get more detailed scene segmentation
    # To reach the total count, we first cluster into (total - edge_count) centers
    num_edges = int(total_count * edge_ratio)
    num_centers = total_count - num_edges

    print(f"Starting hybrid sampling: Plan to extract {num_centers} center samples and {num_edges} edge samples")
    print(f"Running KMeans clustering (clusters: {num_centers}, iterations: {num_em_iter})...")
    
    ret = fastdup.run_kmeans(
        input_dir=input_dir,
        work_dir=work_dir,
        num_clusters=num_centers, 
        num_em_iter=num_em_iter,
        verbose=False
    )

    if ret != 0:
        print("Error: Clustering failed")
        return False

    # 2. Read clustering results
    kmeans_assignments_file = os.path.join(work_dir, "kmeans_assignments.csv")
    if not os.path.exists(kmeans_assignments_file):
        print(f"Error: Clustering result file not found: {kmeans_assignments_file}")
        return False
    
    df_clusters = pd.read_csv(kmeans_assignments_file)
    
    # Ensure each image belongs to only one cluster
    df_clusters = df_clusters.loc[df_clusters.groupby('filename')['distance'].idxmin()]

    # 3. Extract center samples (representative of each cluster)
    df_centers = df_clusters.loc[df_clusters.groupby('cluster')['distance'].idxmin()]
    
    # 4. Extract edge samples
    # Logic: Calculate average distance for each cluster, select the most "scattered" clusters,
    # and take the farthest image from these clusters
    cluster_variance = df_clusters.groupby('cluster')['distance'].mean().sort_values(ascending=False)
    diverse_clusters = cluster_variance.head(num_edges).index.tolist()
    
    # If we don't have enough diverse clusters, use all available clusters
    if len(diverse_clusters) < num_edges:
        diverse_clusters = cluster_variance.index.tolist()[:num_edges]

    edge_list = []
    for c_id in diverse_clusters:
        # Extract the sample with maximum distance to center in this cluster
        cluster_data = df_clusters[df_clusters['cluster'] == c_id]
        if len(cluster_data) > 0:
            edge_sample = cluster_data.nlargest(1, 'distance')
            edge_list.append(edge_sample)
    
    df_edges = pd.concat(edge_list) if edge_list else pd.DataFrame()

    # 5. Merge and export
    category_name = os.path.basename(input_dir.rstrip('/\\'))
    target_dir = os.path.join(output_dir, category_name)
    os.makedirs(target_dir, exist_ok=True)

    copied_centers = 0
    # Copy center samples
    for idx, row in df_centers.iterrows():
        src_path = row['filename']
        if not os.path.exists(src_path):
            continue
        dst_name = f"center_c{row['cluster']}_{os.path.basename(src_path)}"
        shutil.copy2(src_path, os.path.join(target_dir, dst_name))
        copied_centers += 1

    copied_edges = 0
    # Copy edge samples
    for idx, row in df_edges.iterrows():
        src_path = row['filename']
        if not os.path.exists(src_path):
            continue
        dst_name = f"edge_c{row['cluster']}_{os.path.basename(src_path)}"
        shutil.copy2(src_path, os.path.join(target_dir, dst_name))
        copied_edges += 1

    print(f"✓ Hybrid sampling completed! Exported {copied_centers} center + {copied_edges} edge templates to: {target_dir}")
    return True


def process_categories(root_data_dir, output_dir="final_templates", method="center",
                      num_templates=15, edge_ratio=0.2, num_em_iter=30):
    """
    Process multiple category folders in batch
    
    Args:
        root_data_dir: Root directory containing category folders
        output_dir: Output directory for template library
        method: Sampling method ("center" or "hybrid")
        num_templates: Number of templates per category
        edge_ratio: Edge sample ratio (only for hybrid method)
        num_em_iter: Number of EM iterations
    """
    if not os.path.exists(root_data_dir):
        print(f"Error: Root data directory does not exist: {root_data_dir}")
        return
    
    print("=" * 60)
    print(f"Batch processing categories from: {root_data_dir}")
    print(f"Method: {method}, Templates per category: {num_templates}")
    print("=" * 60)
    
    category_folders = [f for f in os.listdir(root_data_dir) 
                       if os.path.isdir(os.path.join(root_data_dir, f))]
    
    if len(category_folders) == 0:
        print(f"Warning: No category folders found in {root_data_dir}")
        return
    
    print(f"Found {len(category_folders)} category folders")
    
    success_count = 0
    for i, category_folder in enumerate(category_folders, 1):
        full_path = os.path.join(root_data_dir, category_folder)
        print(f"\n[{i}/{len(category_folders)}] Processing: {category_folder}")
        
        work_dir = os.path.join(output_dir, f"work_{category_folder}")
        
        if method == "center":
            success = method2_template_sampling(
                input_dir=full_path,
                work_dir=work_dir,
                output_dir=output_dir,
                num_templates=num_templates,
                num_em_iter=num_em_iter
            )
        elif method == "hybrid":
            success = method2_hybrid_sampling(
                input_dir=full_path,
                work_dir=work_dir,
                output_dir=output_dir,
                total_count=num_templates,
                edge_ratio=edge_ratio,
                num_em_iter=num_em_iter
            )
        else:
            print(f"Error: Unknown method '{method}', skipping {category_folder}")
            continue
        
        if success:
            success_count += 1
    
    print("\n" + "=" * 60)
    print(f"Batch processing completed: {success_count}/{len(category_folders)} categories processed successfully")
    print(f"Template library saved to: {output_dir}")
    print("=" * 60)


def GetArgs():
    parser = argparse.ArgumentParser(
        description="Generate template library using fastdup clustering-based sampling for open-set recognition",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Input directory containing category folders (for batch processing) or single category folder")
    parser.add_argument("--output_dir", type=str, default="template_library",
                        help="Output directory for template library")
    parser.add_argument("--method", type=str, choices=["center", "hybrid"], default="center",
                        help="Sampling method: 'center' for cluster center sampling, 'hybrid' for center + edge sampling")
    parser.add_argument("--num_templates", type=int, default=15,
                        help="Number of templates to generate per category (K clusters)")
    parser.add_argument("--edge_ratio", type=float, default=0.2,
                        help="Ratio of edge samples for hybrid method (0.0-1.0, default: 0.2 means 20%%)")
    parser.add_argument("--num_em_iter", type=int, default=30,
                        help="Number of EM iterations for KMeans (default: 30 for complex environments)")
    parser.add_argument("--batch", action="store_true",
                        help="Batch process mode: treat input_dir as root containing multiple category folders")
    parser.add_argument("--single", action="store_true",
                        help="Single category mode: treat input_dir as a single category folder")

    args = parser.parse_args()
    return args


def main():
    """Main function"""
    args = GetArgs()
    
    if not os.path.exists(args.input_dir):
        print(f"Error: Input directory does not exist: {args.input_dir}")
        return
    
    # Determine processing mode
    if args.batch:
        # Batch mode: process all category folders
        process_categories(
            root_data_dir=args.input_dir,
            output_dir=args.output_dir,
            method=args.method,
            num_templates=args.num_templates,
            edge_ratio=args.edge_ratio,
            num_em_iter=args.num_em_iter
        )
    elif args.single:
        # Single category mode
        work_dir = os.path.join(args.output_dir, "work_single")
        
        if args.method == "center":
            method2_template_sampling(
                input_dir=args.input_dir,
                work_dir=work_dir,
                output_dir=args.output_dir,
                num_templates=args.num_templates,
                num_em_iter=args.num_em_iter
            )
        elif args.method == "hybrid":
            method2_hybrid_sampling(
                input_dir=args.input_dir,
                work_dir=work_dir,
                output_dir=args.output_dir,
                total_count=args.num_templates,
                edge_ratio=args.edge_ratio,
                num_em_iter=args.num_em_iter
            )
    else:
        # Auto-detect mode: check if input_dir contains subdirectories
        subdirs = [f for f in os.listdir(args.input_dir) 
                  if os.path.isdir(os.path.join(args.input_dir, f))]
        
        if len(subdirs) > 0:
            # Looks like a root directory with category folders
            print("Auto-detected batch mode: found category subdirectories")
            process_categories(
                root_data_dir=args.input_dir,
                output_dir=args.output_dir,
                method=args.method,
                num_templates=args.num_templates,
                edge_ratio=args.edge_ratio,
                num_em_iter=args.num_em_iter
            )
        else:
            # Single category folder
            print("Auto-detected single category mode")
            work_dir = os.path.join(args.output_dir, "work_single")
            
            if args.method == "center":
                method2_template_sampling(
                    input_dir=args.input_dir,
                    work_dir=work_dir,
                    output_dir=args.output_dir,
                    num_templates=args.num_templates,
                    num_em_iter=args.num_em_iter
                )
            elif args.method == "hybrid":
                method2_hybrid_sampling(
                    input_dir=args.input_dir,
                    work_dir=work_dir,
                    output_dir=args.output_dir,
                    total_count=args.num_templates,
                    edge_ratio=args.edge_ratio,
                    num_em_iter=args.num_em_iter
                )


if __name__ == '__main__':
    main()
