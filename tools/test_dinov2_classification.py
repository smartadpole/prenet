#!/usr/bin/env python3
# encoding: utf-8
'''
@author: 孙昊
@contact: smartadpole@163.com
@file: test_dinov2_classification.py
@time: 2026/1/15 16:19
@desc: Test script for DINOv2 ArcFace model - classify images and visualize by category
'''

import argparse
import csv
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.transforms import functional as TF
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from collections import defaultdict
from train_dinov2_arcface_small import DinoV2Embedder
from train_dinov2_arcface_small import ArcFaceHead
from train_dinov2_arcface_small import build_val_tfm, CenterSquareCrop, make_divisible
from utils.utils import timeit

# Add parent directory to path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

from utils.file import walk_image


# --------- Model Definition (from train_dinov2_arcface_small.py) ----------
def load_model(model_path: str, device: str = "cuda"):
    """
    Load model from checkpoint file.
    
    Args:
        model_path: Path to .pt checkpoint file
        device: Device to load model on
        
    Returns:
        embedder, head, classes, args
    """
    print(f"[Info] Loading model from {model_path}")
    checkpoint = torch.load(model_path, map_location=device)
    
    # Extract model parameters
    args = checkpoint.get("args", {})
    embed_dim = args.get("embed_dim", 256)
    num_classes = args.get("num_classes")
    backbone = args.get("backbone", "dinov2_vitb14")
    img_size = args.get("img_size", 128)
    arc_s = args.get("arc_s", 32.0)
    arc_m = args.get("arc_m", 0.30)
    classes = checkpoint.get("classes", None)
    
    if num_classes is None:
        raise ValueError("Cannot determine num_classes from checkpoint. Please specify --num_classes.")
    
    # Build model
    embedder = DinoV2Embedder(backbone, embed_dim).to(device)
    head = ArcFaceHead(embed_dim, num_classes, s=arc_s, m=arc_m).to(device)
    
    # Load weights
    embedder.load_state_dict(checkpoint["embedder"])
    head.load_state_dict(checkpoint["head"])
    
    embedder.eval()
    head.eval()
    
    print(f"[Info] Model loaded: embed_dim={embed_dim}, num_classes={num_classes}, img_size={img_size}")
    if classes:
        print(f"[Info] Classes: {len(classes)} classes found")
    
    return embedder, head, classes, args


@timeit(100)
@torch.no_grad()
def classify(embedder, head, img_tensor):
    """
    Classify batch of images.
    
    Args:
        embedder: DinoV2Embedder model
        head: ArcFaceHead model
        img_tensor: Tensor of shape [batch_size, C, H, W]
        
    Returns:
        pred_classes: Tensor of shape [batch_size], predicted class indices
        confidences: Tensor of shape [batch_size], confidence scores
        logits: Tensor of shape [batch_size, num_classes], raw logits
    """
    z = embedder(img_tensor)
    W = F.normalize(head.W, dim=1)
    logits = head.s * F.linear(z, W)  # no margin at eval
    probs = F.softmax(logits, dim=1)

    pred_classes = logits.argmax(dim=1)
    confidences = probs.gather(1, pred_classes.unsqueeze(1)).squeeze(1)

    return pred_classes, confidences, logits

@torch.no_grad()
def classify_batch(embedder, head, image_paths: list, transform, device: str):
    """
    Classify a batch of images.
    
    Args:
        embedder: DinoV2Embedder model
        head: ArcFaceHead model
        image_paths: List of image file paths
        transform: Image transform pipeline
        device: Device to run inference on
        
    Returns:
        results: List of tuples (predicted_class, confidence, logits) for each image
                 If an image fails to load, the tuple will be (None, None, None)
    """
    batch_tensors = []
    valid_indices = []
    valid_paths = []
    
    # Load and transform images
    for idx, img_path in enumerate(image_paths):
        try:
            img = Image.open(img_path).convert('RGB')
            img_tensor = transform(img)
            batch_tensors.append(img_tensor)
            valid_indices.append(idx)
            valid_paths.append(img_path)
        except Exception as e:
            print(f"[Warning] Failed to load {img_path}: {e}")
    
    if len(batch_tensors) == 0:
        return [(None, None, None)] * len(image_paths)
    
    # Stack into batch tensor
    batch_tensor = torch.stack(batch_tensors).to(device)
    
    # Classify batch
    pred_classes, confidences, logits = classify(embedder, head, batch_tensor)
    
    # Convert to CPU and numpy for easier handling
    pred_classes = pred_classes.cpu()
    confidences = confidences.cpu()
    logits = logits.cpu()
    
    # Build results list
    results = [(None, None, None)] * len(image_paths)
    for i, valid_idx in enumerate(valid_indices):
        results[valid_idx] = (
            pred_classes[i].item(),
            confidences[i].item(),
            logits[i:i+1]  # Keep as tensor for consistency
        )
    
    return results

@torch.no_grad()
def classify_image(embedder, head, image_path: str, transform, device: str):
    """
    Classify a single image (kept for backward compatibility).
    
    Args:
        embedder: DinoV2Embedder model
        head: ArcFaceHead model
        image_path: Path to image file
        transform: Image transform pipeline
        device: Device to run inference on
        
    Returns:
        predicted_class: int, class index
        confidence: float, confidence score
        logits: torch.Tensor, raw logits
    """
    results = classify_batch(embedder, head, [image_path], transform, device)
    return results[0]


def visualize_by_category(image_results: dict, classes: list = None, output_path: str = None, max_images_per_class: int = 20):
    """
    Visualize images grouped by predicted category. Each class is drawn in a separate figure.
    
    Args:
        image_results: Dict mapping image_path -> (predicted_class, confidence, logits)
        classes: List of class names (optional)
        output_path: Path to save visualization (directory or file path)
        max_images_per_class: Maximum number of images to show per class
    """
    # Group images by predicted class
    class_to_images = defaultdict(list)
    for img_path, (pred_class, confidence, _) in image_results.items():
        if pred_class is not None:
            class_to_images[pred_class].append((img_path, confidence))
    
    # Sort classes by number of images (descending)
    sorted_classes = sorted(class_to_images.items(), key=lambda x: len(x[1]), reverse=True)
    
    print(f"\n[Info] Found {len(sorted_classes)} unique classes")
    for class_idx, images in sorted_classes:
        class_name = classes[class_idx] if classes and class_idx < len(classes) else f"Class_{class_idx}"
        print(f"  Class {class_idx} ({class_name}): {len(images)} images")
    
    # Create visualization
    num_classes = len(sorted_classes)
    if num_classes == 0:
        print("[Warning] No valid predictions to visualize")
        return
    
    # Determine output directory and base filename
    if output_path:
        if os.path.isdir(output_path) or output_path.endswith(os.sep):
            output_dir = output_path if os.path.isdir(output_path) else os.path.dirname(output_path)
            base_filename = "classification_visualization"
        else:
            output_dir = os.path.dirname(output_path) or "."
            base_filename = os.path.splitext(os.path.basename(output_path))[0]
    else:
        output_dir = None
        base_filename = None
    
    # Generate a separate figure for each class
    cols = 4
    saved_paths = []
    
    for class_idx, images in sorted_classes:
        # Limit images per class
        images = sorted(images, key=lambda x: x[1], reverse=True)[:max_images_per_class]
        num_images = len(images)
        rows_for_class = (num_images + cols - 1) // cols  # Ceiling division
        
        if num_images == 0:
            continue
        
        # Create figure for this class
        fig = plt.figure(figsize=(16, min(rows_for_class * 1.5 + 1, 100)))  # Limit max height
        gs = gridspec.GridSpec(rows_for_class + 1, cols, figure=fig, hspace=0.3, wspace=0.2)
        
        # Class title
        class_name = classes[class_idx] if classes and class_idx < len(classes) else f"Class_{class_idx}"
        ax_title = fig.add_subplot(gs[0, :])
        ax_title.axis('off')
        ax_title.text(0.5, 0.5, f"Class {class_idx}: {class_name} ({len(images)} images)", 
                     ha='center', va='center', fontsize=16, fontweight='bold')
        
        # Display images in rows
        for row in range(rows_for_class):
            for col in range(cols):
                img_idx = row * cols + col
                if img_idx < len(images):
                    img_path, confidence = images[img_idx]
                    try:
                        img = Image.open(img_path).convert('RGB')
                        ax = fig.add_subplot(gs[row + 1, col])
                        ax.imshow(img)
                        ax.axis('off')
                        # Truncate filename if too long
                        filename = os.path.basename(img_path)
                        if len(filename) > 20:
                            filename = filename[:17] + "..."
                        ax.set_title(f"{filename}\nConf: {confidence:.3f}", 
                                   fontsize=8, pad=2)
                    except Exception as e:
                        ax = fig.add_subplot(gs[row + 1, col])
                        ax.axis('off')
                        filename = os.path.basename(img_path)
                        if len(filename) > 20:
                            filename = filename[:17] + "..."
                        ax.text(0.5, 0.5, f"Error loading\n{filename}", 
                               ha='center', va='center', fontsize=8, color='red')
                else:
                    # Fill empty cells
                    ax = fig.add_subplot(gs[row + 1, col])
                    ax.axis('off')
        
        plt.suptitle(f"Class {class_idx}: {class_name} - {len(images)} images", 
                    fontsize=18, fontweight='bold', y=0.995)
        
        # Save or show
        if output_path:
            # Generate filename for this class
            safe_class_name = class_name.replace('/', '_').replace('\\', '_').replace(':', '_')
            if output_dir:
                class_output_path = os.path.join(output_dir, f"{base_filename}_class_{class_idx}_{safe_class_name}.png")
            else:
                class_output_path = f"{base_filename}_class_{class_idx}_{safe_class_name}.png"
            
            os.makedirs(os.path.dirname(class_output_path) if os.path.dirname(class_output_path) else ".", exist_ok=True)
            plt.savefig(class_output_path, dpi=150, bbox_inches='tight')
            saved_paths.append(class_output_path)
        else:
            plt.show()
        
        plt.close()
    
    if saved_paths:
        print(f"\n[Info] Generated {len(saved_paths)} visualization files:")
        for path in saved_paths:
            print(f"  - {path}")


def main():
    parser = argparse.ArgumentParser(description="Test DINOv2 ArcFace model on images")
    parser.add_argument("--model_path", type=str, required=True, 
                       help="Path to model checkpoint (.pt file)")
    parser.add_argument("--image_dir", type=str, required=True,
                       help="Directory containing images (supports nested directories)")
    parser.add_argument("--output_dir", type=str, default="test_output",
                       help="Output directory for results")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"],
                       help="Device to run inference on")
    parser.add_argument("--batch_size", "-b", type=int, default=32,
                       help="Batch size for inference")
    parser.add_argument("--max_images_per_class", type=int, default=20,
                       help="Maximum number of images to show per class in visualization")
    parser.add_argument("--num_classes", type=int, default=None,
                       help="Number of classes (if not in checkpoint)")
    args = parser.parse_args()

    name = os.path.basename(os.path.dirname(args.model_path))
    output_dir = os.path.join(args.output_dir, name)
    
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
        print("[Warning] CUDA not available, using CPU")
    
    # Load model
    embedder, head, classes, model_args = load_model(args.model_path, device)
    
    # Override num_classes if provided
    if args.num_classes is not None:
        print(f"[Info] Overriding num_classes to {args.num_classes}")
    
    # Build transform
    img_size = model_args.get("img_size", 128)
    transform = build_val_tfm(img_size)
    
    # Collect all images
    print(f"\n[Info] Scanning images in {args.image_dir}")
    image_paths = walk_image(args.image_dir)
    print(f"[Info] Found {len(image_paths)} images")
    
    if len(image_paths) == 0:
        print("[Error] No images found in the specified directory")
        return
    
    # Classify images in batches
    print(f"\n[Info] Classifying images with batch_size={args.batch_size}...")
    image_results = {}
    results_by_class = defaultdict(list)
    
    # Process images in batches
    num_batches = (len(image_paths) + args.batch_size - 1) // args.batch_size
    with tqdm(total=len(image_paths), desc="Processing") as pbar:
        for batch_idx in range(num_batches):
            start_idx = batch_idx * args.batch_size
            end_idx = min(start_idx + args.batch_size, len(image_paths))
            batch_paths = image_paths[start_idx:end_idx]
            
            # Classify batch
            batch_results = classify_batch(embedder, head, batch_paths, transform, device)
            
            # Process results
            for img_path, (pred_class, confidence, logits) in zip(batch_paths, batch_results):
                if pred_class is not None:
                    image_results[img_path] = (pred_class, confidence, logits)
                    class_name = classes[pred_class] if classes and pred_class < len(classes) else f"Class_{pred_class}"
                    results_by_class[pred_class].append((img_path, confidence))
                pbar.update(1)
    
    # Print summary
    print(f"\n[Info] Successfully classified {len(image_results)}/{len(image_paths)} images")
    
    # Save results as CSV
    os.makedirs(output_dir, exist_ok=True)
    results_file = os.path.join(output_dir, "classification_results.csv")
    with open(results_file, 'w', encoding='utf-8-sig', newline='') as f:
        writer = csv.writer(f)
        # Write header
        writer.writerow(['image_path', 'class_index', 'class_name', 'confidence'])
        # Write data rows
        for class_idx in sorted(results_by_class.keys()):
            class_name = classes[class_idx] if classes and class_idx < len(classes) else f"Class_{class_idx}"
            for img_path, confidence in sorted(results_by_class[class_idx], key=lambda x: x[1], reverse=True):
                writer.writerow([img_path, class_idx, class_name, f"{confidence:.4f}"])
    
    print(f"[Info] CSV results saved to {results_file}")
    
    # Visualize results
    print("\n[Info] Generating visualization...")
    vis_output_path = os.path.join(output_dir, "classification_visualization.png")
    visualize_by_category(image_results, classes, vis_output_path, args.max_images_per_class)
    
    print("\n[Info] Test completed!")


if __name__ == "__main__":
    main()
