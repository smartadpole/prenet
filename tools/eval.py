#!/usr/bin/env python3
# encoding: utf-8
'''
@author: 孙昊
@contact: smartadpole@163.com
@file: eval.py
@time: 2026/1/16 14:50
@desc: Evaluation script for DINOv2 ArcFace model - calculate accuracy metrics and visualize results
'''
import sys
import os

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

import argparse
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from collections import defaultdict, Counter
from train_dinov2_arcface_small import DinoV2Embedder
from train_dinov2_arcface_small import ArcFaceHead
from train_dinov2_arcface_small import build_val_tfm, CenterSquareCrop
from utils.utils import timeit

plt.rcParams['font.sans-serif'] = ['SimHei', 'WenQuanYi Micro Hei', 'Noto Sans CJK SC', 'Droid Sans Fallback']
plt.rcParams['axes.unicode_minus'] = False


def load_test_file(test_file: str):
    """
    Load test data from txt file.
    
    File format: Three columns separated by space or tab:
        - Column 1: Absolute path to image
        - Column 2: Label ID (integer)
        - Column 3: Class name (string)
    
    Args:
        test_file: Path to test file (.txt)
        
    Returns:
        test_data: List of tuples (image_path, label_id, class_name)
    """
    test_data = []
    
    if not os.path.exists(test_file):
        raise FileNotFoundError(f"Test file not found: {test_file}")
    
    with open(test_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            
            # Try to split by tab first, then by space (split max 3 times to handle paths with spaces)
            parts = line.split('\t') if '\t' in line else line.split(',') if ',' in line else line.split(' ')
            
            if len(parts) < 3:
                print(f"[Warning] Line {line_num} has less than 3 columns, skipping: {line}")
                continue
            
            image_path = parts[0].strip()
            try:
                label_id = int(parts[1].strip())
            except ValueError:
                print(f"[Warning] Line {line_num} has invalid label ID, skipping: {line}")
                continue
            class_name = parts[2].strip()
            
            test_data.append((image_path, label_id, class_name))
    
    print(f"[Info] Loaded {len(test_data)} test samples from {test_file}")
    return test_data


def load_model(model_path: str, device: str = "cuda"):
    """
    Load model from checkpoint file.
    
    Args:
        model_path: Path to .pt checkpoint file
        device: Device to load model on
        
    Returns:
        embedder, head, args
    """
    print(f"[Info] Loading model from {model_path}")
    checkpoint = torch.load(model_path, map_location=device)
    
    args = checkpoint.get("args", {})
    embed_dim = args.get("embed_dim", 256)
    num_classes = args.get("num_classes")
    backbone = args.get("backbone", "dinov2_vitb14")
    img_size = args.get("img_size", 128)
    arc_s = args.get("arc_s", 32.0)
    arc_m = args.get("arc_m", 0.30)
    
    if num_classes is None:
        raise ValueError("Cannot determine num_classes from checkpoint. Please specify --num_classes.")
    
    embedder = DinoV2Embedder(backbone, embed_dim, train=False).to(device)
    head = ArcFaceHead(embed_dim, num_classes, s=arc_s, m=arc_m).to(device)
    
    embedder.load_state_dict(checkpoint["embedder"])
    head.load_state_dict(checkpoint["head"])
    
    embedder.eval()
    head.eval()
    
    print(f"[Info] Model loaded: embed_dim={embed_dim}, num_classes={num_classes}, img_size={img_size}")
    
    return embedder, head, args


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
    """
    z = embedder(img_tensor)
    W = F.normalize(head.W, dim=1)
    logits = head.s * F.linear(z, W)
    probs = F.softmax(logits, dim=1)
    
    pred_classes = logits.argmax(dim=1)
    confidences = probs.gather(1, pred_classes.unsqueeze(1)).squeeze(1)
    
    return pred_classes, confidences


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
        results: List of tuples (predicted_class, confidence) for each image
                 If an image fails to load, the tuple will be (None, None)
    """
    batch_tensors = []
    valid_indices = []
    valid_paths = []
    
    for idx, img_path in enumerate(image_paths):
        try:
            if not os.path.exists(img_path):
                print(f"[Warning] Image not found: {img_path}")
                continue
            img = Image.open(img_path).convert('RGB')
            img_tensor = transform(img)
            batch_tensors.append(img_tensor)
            valid_indices.append(idx)
            valid_paths.append(img_path)
        except Exception as e:
            print(f"[Warning] Failed to load {img_path}: {e}")
    
    if len(batch_tensors) == 0:
        return [(None, None)] * len(image_paths)
    
    batch_tensor = torch.stack(batch_tensors).to(device)
    pred_classes, confidences = classify(embedder, head, batch_tensor)
    
    pred_classes = pred_classes.cpu()
    confidences = confidences.cpu()
    
    results = [(None, None)] * len(image_paths)
    for i, valid_idx in enumerate(valid_indices):
        results[valid_idx] = (
            pred_classes[i].item(),
            confidences[i].item()
        )
    
    return results


def calculate_metrics(predictions, ground_truth):
    """
    Calculate overall accuracy and per-class accuracy with counts.
    
    Args:
        predictions: List of predicted class IDs
        ground_truth: List of ground truth class IDs
        
    Returns:
        overall_accuracy: float, overall accuracy
        overall_correct: int, total number of correct predictions
        overall_wrong: int, total number of wrong predictions
        overall_total: int, total number of valid samples
        per_class_accuracy: dict, mapping class_id -> accuracy
        per_class_counts: dict, mapping class_id -> (correct, wrong, total)
        confusion_data: dict, mapping (true_label, pred_label) -> count
    """
    assert len(predictions) == len(ground_truth)
    
    # Overall accuracy
    correct = sum(1 for p, g in zip(predictions, ground_truth) if p == g and p is not None)
    total = sum(1 for p, g in zip(predictions, ground_truth) if p is not None and g is not None)
    wrong = total - correct
    overall_accuracy = correct / total if total > 0 else 0.0
    
    # Per-class accuracy and counts
    per_class_correct = defaultdict(int)
    per_class_total = defaultdict(int)
    confusion_data = defaultdict(int)
    
    for pred, true_label in zip(predictions, ground_truth):
        if pred is None or true_label is None:
            continue
        per_class_total[true_label] += 1
        confusion_data[(true_label, pred)] += 1
        if pred == true_label:
            per_class_correct[true_label] += 1
    
    per_class_accuracy = {
        class_id: per_class_correct[class_id] / per_class_total[class_id]
        if per_class_total[class_id] > 0 else 0.0
        for class_id in per_class_total.keys()
    }
    
    per_class_counts = {
        class_id: (
            per_class_correct[class_id],
            per_class_total[class_id] - per_class_correct[class_id],
            per_class_total[class_id]
        )
        for class_id in per_class_total.keys()
    }
    
    return overall_accuracy, correct, wrong, total, per_class_accuracy, per_class_counts, dict(confusion_data)


def visualize_results(correct_samples, wrong_samples, classes_dict, per_class_accuracy, per_class_counts, output_path: str):
    """
    Visualize evaluation results by class: 10 correct samples and all wrong samples per class.
    
    Args:
        correct_samples: List of tuples (image_path, true_label, pred_label, confidence, class_name)
        wrong_samples: List of tuples (image_path, true_label, pred_label, confidence, true_name, pred_name)
        classes_dict: Dict mapping label_id -> class_name
        per_class_accuracy: Dict mapping class_id -> accuracy (float)
        per_class_counts: Dict mapping class_id -> (correct, wrong, total)
        output_path: Path to save visualization
    """
    saved_paths = []
    
    # Group samples by true label (class)
    correct_by_class = defaultdict(list)
    wrong_by_class = defaultdict(list)
    
    for sample in correct_samples:
        img_path, true_label, pred_label, confidence, class_name = sample
        correct_by_class[true_label].append(sample)
    
    for sample in wrong_samples:
        img_path, true_label, pred_label, confidence, true_name, pred_name = sample
        wrong_by_class[true_label].append(sample)
    
    # Get all classes that have samples
    all_classes = set(correct_by_class.keys()) | set(wrong_by_class.keys())
    
    # Generate one visualization per class
    for class_id in sorted(all_classes):
        class_name = classes_dict.get(class_id, f"Class_{class_id}")
        class_correct = correct_by_class.get(class_id, [])
        class_wrong = wrong_by_class.get(class_id, [])
        
        # Get accuracy and counts for this class
        acc = per_class_accuracy.get(class_id, 0.0)
        correct_count, wrong_count, total_count = per_class_counts.get(class_id, (0, 0, 0))
        
        # Sort correct samples by confidence and take top 10
        class_correct_sorted = sorted(class_correct, key=lambda x: x[3], reverse=True)[:10]
        
        # Combine: correct samples first, then wrong samples
        all_samples = []
        
        # Add correct samples with type marker
        for sample in class_correct_sorted:
            all_samples.append(('correct', sample))
        
        # Add wrong samples with type marker
        for sample in class_wrong:
            all_samples.append(('wrong', sample))
        
        if len(all_samples) == 0:
            continue
        
        # Create visualization
        cols = 5
        rows = (len(all_samples) + cols - 1) // cols
        
        fig = plt.figure(figsize=(20, min(rows * 4 + 1, 200)))
        gs = gridspec.GridSpec(rows, cols, figure=fig, hspace=0.4, wspace=0.2)

        # Display samples
        for idx, (sample_type, sample) in enumerate(all_samples):
            row = idx // cols
            col = idx % cols
            img_path = None
            
            try:
                if sample_type == 'correct':
                    img_path, true_label, pred_label, confidence, class_name = sample
                    img = Image.open(img_path).convert('RGB')
                    ax = fig.add_subplot(gs[row, col])
                    ax.imshow(img)
                    ax.axis('off')
                    ax.set_title(f"[Correct]\nConf: {confidence:.3f}",
                                fontsize=9, pad=3, color='green')
                else:  # wrong
                    img_path, true_label, pred_label, confidence, true_name, pred_name = sample
                    img = Image.open(img_path).convert('RGB')
                    ax = fig.add_subplot(gs[row, col])
                    ax.imshow(img)
                    ax.axis('off')
                    ax.set_title(f"[Wrong]\nTrue: {true_name}\nPred: {pred_name}\nConf: {confidence:.3f}",
                                fontsize=8, pad=3, color='red')
            except Exception as e:
                ax = fig.add_subplot(gs[row, col])
                ax.axis('off')
                if img_path:
                    filename = os.path.basename(img_path)
                else:
                    filename = "Unknown"
                ax.text(0.5, 0.5, f"Error\n{filename}",
                       ha='center', va='center', fontsize=8, color='red')
        
        # Fill empty cells
        for idx in range(len(all_samples), rows * cols):
            row = idx // cols
            col = idx % cols
            ax = fig.add_subplot(gs[row, col])
            ax.axis('off')
        
        suptitle_text = f"Class {class_id}: {class_name} | Acc: {acc:.4f} ({acc*100:.2f}%) | Wrong: {wrong_count} | Total: {total_count}"
        plt.suptitle(suptitle_text,
                    fontsize=18, fontweight='bold', y=0.995)
        
        # Generate output filename
        safe_class_name = class_name.replace('/', '_').replace('\\', '_').replace(':', '_')
        output_dir = os.path.dirname(output_path) if os.path.dirname(output_path) else "."
        base_filename = os.path.splitext(os.path.basename(output_path))[0]
        class_output_path = os.path.join(output_dir, f"class_{class_id}_{safe_class_name}.png")
        
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(class_output_path, dpi=150, bbox_inches='tight')
        saved_paths.append(class_output_path)
        plt.close()
    
    if saved_paths:
        print(f"\n[Info] Generated {len(saved_paths)} visualization files:")
        for path in saved_paths:
            print(f"  - {path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate DINOv2 ArcFace model accuracy",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to model checkpoint (.pt file)")
    parser.add_argument("--test_file", type=str, required=True,
                        help="Path to test file (.txt). Format: image_path label_id class_name")
    parser.add_argument("--output_dir", type=str, default="eval_output",
                        help="Output directory for results")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"],
                        help="Device to run inference on")
    parser.add_argument("--batch_size", "-b", type=int, default=32,
                        help="Batch size for inference")
    parser.add_argument("--num_classes", type=int, default=None,
                        help="Number of classes (if not in checkpoint)")
    args = parser.parse_args()

    output_dir = args.output_dir + "_eval" + os.path.basename(os.path.dirname(args.model_path))
    # Setup device
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
        print("[Warning] CUDA not available, using CPU")
    
    # Load test data
    test_data = load_test_file(args.test_file)
    if len(test_data) == 0:
        print("[Error] No valid test samples found")
        return
    
    # Build class name dict
    classes_dict = {}
    for _, label_id, class_name in test_data:
        if label_id not in classes_dict:
            classes_dict[label_id] = class_name
    
    # Load model
    embedder, head, model_args = load_model(args.model_path, device)
    
    # Override num_classes if provided
    if args.num_classes is not None:
        print(f"[Warning] Overriding num_classes is not supported during evaluation")
    
    # Build transform
    img_size = model_args.get("img_size", 128)
    transform = build_val_tfm(img_size)
    
    # Extract image paths and labels
    image_paths = [item[0] for item in test_data]
    true_labels = [item[1] for item in test_data]
    class_names = [item[2] for item in test_data]
    
    # Classify images in batches
    print(f"\n[Info] Classifying {len(image_paths)} images with batch_size={args.batch_size}...")
    predictions = []
    confidences = []
    
    num_batches = (len(image_paths) + args.batch_size - 1) // args.batch_size
    with tqdm(total=len(image_paths), desc="Processing") as pbar:
        for batch_idx in range(num_batches):
            start_idx = batch_idx * args.batch_size
            end_idx = min(start_idx + args.batch_size, len(image_paths))
            batch_paths = image_paths[start_idx:end_idx]
            
            batch_results = classify_batch(embedder, head, batch_paths, transform, device)
            
            for pred_class, confidence in batch_results:
                predictions.append(pred_class)
                confidences.append(confidence)
            
            pbar.update(len(batch_paths))
    
    # Calculate metrics
    print("\n[Info] Calculating metrics...")
    (overall_accuracy, overall_correct, overall_wrong, overall_total,
     per_class_accuracy, per_class_counts, confusion_data) = calculate_metrics(predictions, true_labels)
    
    # Print results
    print(f"\n{'='*60}")
    print(f"Evaluation Results")
    print(f"{'='*60}")
    print(f"Overall Accuracy: {overall_accuracy:.4f} ({overall_accuracy*100:.2f}%)")
    print(f"Overall: Correct={overall_correct}, Wrong={overall_wrong}, Total={overall_total}")
    print(f"\nPer-Class Accuracy:")
    for class_id in sorted(per_class_accuracy.keys()):
        class_name = classes_dict.get(class_id, f"Class_{class_id}")
        acc = per_class_accuracy[class_id]
        correct_count, wrong_count, total_count = per_class_counts[class_id]
        print(f"  Class {class_id} ({class_name}): "
              f"Accuracy={acc*100:.2f}%, "
              f"({correct_count} / {total_count})")
    print(f"{'='*60}\n")
    
    # Prepare samples for visualization
    correct_samples = []
    wrong_samples = []
    
    for i, (img_path, true_label, pred_label, confidence, class_name) in enumerate(
        zip(image_paths, true_labels, predictions, confidences, class_names)
    ):
        if pred_label is None:
            continue
        
        if pred_label == true_label:
            correct_samples.append((img_path, true_label, pred_label, confidence, class_name))
        else:
            pred_name = classes_dict.get(pred_label, f"Class_{pred_label}")
            wrong_samples.append((img_path, true_label, pred_label, confidence, class_name, pred_name))
    
    print(f"[Info] Correct predictions: {len(correct_samples)}")
    print(f"[Info] Wrong predictions: {len(wrong_samples)}")
    
    # Save results to file
    os.makedirs(output_dir, exist_ok=True)
    results_file = os.path.join(output_dir, "evaluation_results.txt")
    with open(results_file, 'w', encoding='utf-8') as f:
        f.write("Evaluation Results\n")
        f.write("="*60 + "\n")
        f.write(f"Overall Accuracy: {overall_accuracy:.4f} ({overall_accuracy*100:.2f}%)\n")
        f.write(f"Overall: Correct={overall_correct}, Wrong={overall_wrong}, Total={overall_total}\n\n")
        f.write("Per-Class Accuracy:\n")
        for class_id in sorted(per_class_accuracy.keys()):
            class_name = classes_dict.get(class_id, f"Class_{class_id}")
            acc = per_class_accuracy[class_id]
            correct_count, wrong_count, total_count = per_class_counts[class_id]
            f.write(f"  Class {class_id} ({class_name}): "
                    f"Accuracy={acc*100:.2f}%, "
                    f"({correct_count} / {total_count})\n")
        f.write("\n" + "="*60 + "\n")
    
    print(f"[Info] Results saved to {results_file}")
    
    # Visualize results
    if correct_samples or wrong_samples:
        print("\n[Info] Generating visualizations...")
        vis_output_path = os.path.join(output_dir, "evaluation_visualization.png")
        visualize_results(correct_samples, wrong_samples, classes_dict, per_class_accuracy, per_class_counts, vis_output_path)
    
    print("\n[Info] Evaluation completed!")


if __name__ == '__main__':
    main()
