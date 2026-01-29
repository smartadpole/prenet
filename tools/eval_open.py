#!/usr/bin/env python3
# encoding: utf-8
'''
@author: 孙昊
@contact: smartadpole@163.com
@file: eval_open.py
@time: 2026/01/26 16:40
@desc: Open-set recognition evaluator using offline feature gallery and metric learning
'''
import sys
import os

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

# Add tools directory to path for importing eval module
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

import argparse
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from collections import defaultdict
import numpy as np
import hashlib
from train_dinov2_arcface_small import DinoV2Embedder, build_val_tfm
from utils.logger import logger_manager

# Import from eval.py in the same directory
import eval as eval_module
load_test_file = eval_module.load_test_file
visualize_results = eval_module.visualize_results

IMG_EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')

logger_manager.set_log_level(level="info")


def _collect_images(root_dir):
    image_paths = []
    for root, _, files in os.walk(root_dir):
        for fname in files:
            if fname.lower().endswith(IMG_EXTS):
                image_paths.append(os.path.join(root, fname))
    return image_paths


def _compute_template_fingerprint(template_data):
    hasher = hashlib.sha1()
    for img_path, label_id, class_name in template_data:
        hasher.update(str(img_path).encode('utf-8'))
        hasher.update(b'\0')
        hasher.update(str(label_id).encode('utf-8'))
        hasher.update(b'\0')
        hasher.update(str(class_name).encode('utf-8'))
        hasher.update(b'\0')
        try:
            mtime = os.path.getmtime(img_path)
        except OSError:
            mtime = 0
        hasher.update(str(mtime).encode('utf-8'))
        hasher.update(b'\n')
    return hasher.hexdigest()


def _make_cache_key(template_fingerprint, model_path, outlier_threshold, img_size):
    raw_key = f"{template_fingerprint}|{os.path.abspath(model_path)}|{outlier_threshold}|{img_size}"
    return hashlib.sha1(raw_key.encode('utf-8')).hexdigest()


def _ensure_dir(path):
    if path:
        os.makedirs(path, exist_ok=True)


def _resolve_cache_dir(template_path, model_path):
    if os.path.isdir(template_path):
        template_parent = os.path.dirname(template_path)
    else:
        template_parent = os.path.dirname(os.path.abspath(template_path))
    model_version = os.path.basename(os.path.dirname(model_path))
    return os.path.join(template_parent, f"gallery_cache_{model_version}")


def _load_gallery_cache(cache_path):
    if not os.path.exists(cache_path):
        return None
    try:
        return torch.load(cache_path, map_location="cpu")
    except Exception as e:
        print(f"加载模板缓存失败，忽略缓存: {cache_path} ({e})", level="warning")
        return None


def _save_gallery_cache(cache_path, cache_payload):
    try:
        _ensure_dir(os.path.dirname(cache_path))
        torch.save(cache_payload, cache_path)
    except Exception as e:
        print(f"保存模板缓存失败: {cache_path} ({e})", level="warning")


def _prepare_visualization_samples(results):
    correct_samples = []
    wrong_samples = []
    for result in results:
        if result['pred_label'] is None:
            wrong_samples.append((
                result['path'],
                result['true_label'],
                None,
                result['confidence'],
                result['true_name'],
                "Unknown/Rejected"
            ))
            continue
        if result['pred_label'] == result['true_label']:
            correct_samples.append((
                result['path'],
                result['true_label'],
                result['pred_label'],
                result['confidence'],
                result['true_name']
            ))
        else:
            wrong_samples.append((
                result['path'],
                result['true_label'],
                result['pred_label'],
                result['confidence'],
                result['true_name'],
                result['pred_name']
            ))
    return correct_samples, wrong_samples


def _write_evaluation_results(results_file, metrics):
    with open(results_file, 'w', encoding='utf-8') as f:
        f.write("Evaluation Results\n")
        f.write("="*60 + "\n")
        f.write(f"Overall Accuracy: {metrics['overall_accuracy']:.4f} "
                f"({metrics['overall_accuracy']*100:.2f}%)\n")
        f.write(f"Overall: Correct={metrics['overall_correct']}, "
                f"Wrong={metrics['overall_wrong']}, "
                f"Total={metrics['overall_total']}\n\n")
        f.write("Per-Class Accuracy:\n")
        for class_name in sorted(metrics['per_class_accuracy'].keys()):
            acc = metrics['per_class_accuracy'][class_name]
            correct_count, wrong_count, total_count = metrics['per_class_counts'][class_name]
            f.write(f"  Label {class_name}: "
                    f"Accuracy={acc*100:.2f}%, "
                    f"({correct_count} / {total_count})\n")
        f.write("\n" + "="*60 + "\n")


def _save_visualizations(output_dir, results, metrics, save_vis):
    if not save_vis:
        return

    print(f"正在保存可视化结果...", level="info")
    correct_samples, wrong_samples = _prepare_visualization_samples(results)
    if not correct_samples and not wrong_samples:
        return
    vis_output_path = os.path.join(output_dir, "evaluation_visualization.png")
    visualize_results(
        correct_samples,
        wrong_samples,
        metrics['classes_dict'],
        metrics['per_class_accuracy'],
        metrics['per_class_counts'],
        vis_output_path
    )


def load_template_data(template_path):
    if not os.path.exists(template_path):
        raise FileNotFoundError(f"Template path not found: {template_path}")

    if os.path.isfile(template_path):
        ext = os.path.splitext(template_path)[1].lower()
        if ext in ('.txt', '.csv'):
            template_data = load_test_file(template_path)
            normalized = []
            for img_path, label_id, class_name in template_data:
                normalized.append((os.path.abspath(img_path), label_id, class_name))
            return normalized
        raise ValueError(f"Unsupported template file extension: {ext}")

    subdirs = sorted(
        d for d in os.listdir(template_path)
        if os.path.isdir(os.path.join(template_path, d))
    )
    template_data = []
    label_map = {name: idx for idx, name in enumerate(subdirs)}
    for class_name in subdirs:
        class_dir = os.path.join(template_path, class_name)
        images = _collect_images(class_dir)
        if not images:
            print(f"模板目录无有效图片，跳过: {class_dir}", level="warning")
            continue
        label_id = label_map[class_name]
        for img_path in images:
            template_data.append((os.path.abspath(img_path), label_id, class_name))

    if not template_data:
        raise ValueError("模板目录为空，无法构建特征库")
    return template_data


class OpenSetEvaluator:
    """
    Open-set recognition evaluator using offline feature gallery.
    
    This class implements metric learning-based open-set recognition:
    1. Builds a feature gallery from template images
    2. Performs Top-K nearest neighbor search for query images
    3. Uses weighted voting to handle class imbalance
    4. Applies threshold-based open-set rejection
    """
    
    def __init__(self, model_path, device="cuda"):
        """
        Initialize the open-set evaluator.
        
        Args:
            model_path: Path to model checkpoint (.pt file)
            device: Device to run inference on
        """
        self.device = device
        self.model_path = model_path
        if device == "cuda" and not torch.cuda.is_available():
            self.device = "cpu"
            print("CUDA 不可用，使用 CPU", level="warning")
        
        self.embedder, self.img_size = self._load_model(model_path)
        self.transform = build_val_tfm(self.img_size)
        self.gallery_features = None
        self.gallery_labels = None
        self.label_to_name = {}
        self.class_sample_counts = defaultdict(int)
        self.last_metrics = None

    def _reset_gallery_state(self):
        self.gallery_features = None
        self.gallery_labels = None
        self.label_to_name = {}
        self.class_sample_counts = defaultdict(int)

    def _load_model(self, model_path):
        """
        Load DinoV2Embedder from checkpoint.
        
        Args:
            model_path: Path to checkpoint file
            
        Returns:
            embedder: DinoV2Embedder model
            img_size: Image size used during training
        """
        checkpoint = torch.load(model_path, map_location=self.device)
        args = checkpoint.get("args", {})
        
        backbone = args.get("backbone", "dinov2_vitb14")
        embed_dim = args.get("embed_dim", 256)
        img_size = args.get("img_size", 128)
        
        embedder = DinoV2Embedder(backbone, embed_dim, train=False)
        embedder.load_state_dict(checkpoint["embedder"])
        embedder.to(self.device).eval()
        
        return embedder, img_size
    
    @torch.no_grad()
    def _build_gallery_from_data(self, template_data, outlier_threshold=2.0):
        if len(template_data) == 0:
            raise ValueError("模板文件为空，无法构建特征库")

        self._reset_gallery_state()
        features_by_class = defaultdict(list)

        for img_path, label_id, class_name in tqdm(template_data, desc="提取模板特征"):
            if label_id not in self.label_to_name:
                self.label_to_name[label_id] = class_name

            try:
                if not os.path.exists(img_path):
                    print(f"图片不存在，跳过: {img_path}", level="warning")
                    continue

                img = Image.open(img_path).convert('RGB')
                img_tensor = self.transform(img).unsqueeze(0).to(self.device)
                feat = self.embedder(img_tensor)
                features_by_class[label_id].append(feat.cpu())
                self.class_sample_counts[label_id] += 1
            except Exception as e:
                print(f"处理图片失败 {img_path}: {e}", level="warning")
                continue

        all_features = []
        all_labels = []

        for label_id in features_by_class.keys():
            class_features = torch.cat(features_by_class[label_id], dim=0)
            valid_features = self._remove_outliers(class_features, outlier_threshold)
            all_features.append(valid_features)
            all_labels.extend([label_id] * len(valid_features))

        if len(all_features) == 0:
            raise ValueError("特征库构建失败，没有有效特征")

        self.gallery_features = torch.cat(all_features, dim=0).to(self.device)
        self.gallery_labels = torch.tensor(all_labels).to(self.device)


    @torch.no_grad()
    def build_gallery(self, template_path, outlier_threshold=2.0):
        """
        Build feature gallery from template images.
        
        Args:
            template_path: Path to template file or directory
            outlier_threshold: Threshold for outlier removal (in standard deviations)
        """
        template_data = load_template_data(template_path)
        self._build_gallery_from_data(template_data, outlier_threshold=outlier_threshold)

    @torch.no_grad()
    def build_gallery_cached(self, template_path, outlier_threshold=2.0):
        template_data = load_template_data(template_path)
        template_fingerprint = _compute_template_fingerprint(template_data)
        cache_key = _make_cache_key(template_fingerprint, self.model_path, outlier_threshold, self.img_size)
        cache_dir = _resolve_cache_dir(template_path, self.model_path)
        cache_path = os.path.join(cache_dir, f"gallery_cache_{cache_key}.pt")

        cache_payload = _load_gallery_cache(cache_path)
        if cache_payload and cache_payload.get("template_fingerprint") == template_fingerprint:
            self._reset_gallery_state()
            self.gallery_features = cache_payload["gallery_features"].to(self.device)
            self.gallery_labels = cache_payload["gallery_labels"].to(self.device)
            self.label_to_name = cache_payload["label_to_name"]
            self.class_sample_counts = defaultdict(int, cache_payload["class_sample_counts"])
            return

        self._build_gallery_from_data(template_data, outlier_threshold=outlier_threshold)
        cache_payload = {
            "template_fingerprint": template_fingerprint,
            "gallery_features": self.gallery_features.cpu(),
            "gallery_labels": self.gallery_labels.cpu(),
            "label_to_name": self.label_to_name,
            "class_sample_counts": dict(self.class_sample_counts)
        }
        _save_gallery_cache(cache_path, cache_payload)
    
    def _remove_outliers(self, features, threshold=2.0):
        """
        Remove outliers from class features using distance-based filtering.
        
        Args:
            features: Tensor of shape [N, D] containing features for one class
            threshold: Threshold in standard deviations for outlier removal
            
        Returns:
            valid_features: Tensor of shape [M, D] with outliers removed (M <= N)
        """
        if len(features) <= 1:
            return features
        
        # Compute mean feature
        mean_feat = features.mean(dim=0, keepdim=True)
        mean_feat = F.normalize(mean_feat, dim=1)
        
        # Compute cosine distances to mean
        similarities = torch.mm(features, mean_feat.t()).squeeze(1)
        distances = 1 - similarities
        
        # Compute statistics
        mean_dist = distances.mean().item()
        std_dist = distances.std().item()
        
        if std_dist < 1e-6:
            return features
        
        # Filter outliers
        cutoff = mean_dist + threshold * std_dist
        valid_mask = distances <= cutoff
        
        if valid_mask.sum() == 0:
            return features
        
        return features[valid_mask]

    def _load_image_tensors(self, image_paths):
        batch_tensors = []
        valid_indices = []
        for idx, img_path in enumerate(image_paths):
            try:
                if not os.path.exists(img_path):
                    print(f"图片不存在，跳过: {img_path}", level="warning")
                    continue
                img = Image.open(img_path).convert('RGB')
                img_tensor = self.transform(img)
                batch_tensors.append(img_tensor)
                valid_indices.append(idx)
            except Exception as e:
                print(f"加载图片失败 {img_path}: {e}", level="warning")
        return batch_tensors, valid_indices

    def _predict_batch(self, image_paths, top_k=5, threshold=0.6, margin_threshold=0.1):
        if self.gallery_features is None:
            raise ValueError("特征库未构建，请先调用 build_gallery()")

        batch_tensors, valid_indices = self._load_image_tensors(image_paths)
        results = [(-1, 0.0) for _ in image_paths]
        if not batch_tensors:
            return results

        batch_tensor = torch.stack(batch_tensors).to(self.device)
        query_feats = self.embedder(batch_tensor)

        sims = torch.mm(query_feats, self.gallery_features.t())
        k = min(top_k, sims.shape[1])
        vals, indices = torch.topk(sims, k, dim=1)

        for row_idx, original_idx in enumerate(valid_indices):
            row_vals = vals[row_idx]
            row_indices = indices[row_idx]
            vote_scores = self._weighted_voting(row_vals, row_indices)
            if not vote_scores:
                results[original_idx] = (-1, 0.0)
                continue

            sorted_classes = sorted(vote_scores.items(), key=lambda x: x[1], reverse=True)
            best_label, best_score = sorted_classes[0]
            max_sim = row_vals[0].item()

            if max_sim < threshold:
                results[original_idx] = (-1, max_sim)
                continue

            if len(sorted_classes) > 1:
                second_score = sorted_classes[1][1]
                margin = best_score - second_score
                if margin < margin_threshold:
                    results[original_idx] = (-1, max_sim)
                    continue

            results[original_idx] = (best_label, max_sim)

        return results
    
    @torch.no_grad()
    def identify(self, img_path, top_k=5, threshold=0.6, margin_threshold=0.1):
        """
        Identify image using open-set recognition.
        
        Args:
            img_path: Path to query image
            top_k: Number of nearest neighbors to consider
            threshold: Absolute similarity threshold for open-set rejection
            margin_threshold: Margin threshold between top-1 and top-2 scores
            
        Returns:
            pred_label: Predicted label ID (-1 for unknown)
            confidence: Confidence score (max similarity)
            pred_name: Predicted class name
        """
        if self.gallery_features is None:
            raise ValueError("特征库未构建，请先调用 build_gallery()")
        
        try:
            img = Image.open(img_path).convert('RGB')
            img_tensor = self.transform(img).unsqueeze(0).to(self.device)
            query_feat = self.embedder(img_tensor)  # Already L2 normalized
        except Exception as e:
            print(f"加载图片失败 {img_path}: {e}", level="warning")
            return -1, 0.0, "未知"
        
        # Compute cosine similarities (dot product since features are normalized)
        similarities = torch.mm(query_feat, self.gallery_features.t()).squeeze(0)
        
        # Get Top-K nearest neighbors
        k = min(top_k, len(similarities))
        vals, indices = torch.topk(similarities, k)
        
        # Weighted voting with class imbalance handling
        vote_scores = self._weighted_voting(vals, indices)
        
        if not vote_scores:
            return -1, 0.0, "未知"
        
        # Get top-2 classes for margin check
        sorted_classes = sorted(vote_scores.items(), key=lambda x: x[1], reverse=True)
        best_label, best_score = sorted_classes[0]
        max_sim = vals[0].item()
        
        # Open-set rejection: absolute threshold
        if max_sim < threshold:
            return -1, max_sim, "未知"
        
        # Open-set rejection: margin threshold (if more than one class)
        if len(sorted_classes) > 1:
            second_score = sorted_classes[1][1]
            margin = best_score - second_score
            if margin < margin_threshold:
                return -1, max_sim, "未知"
        
        return best_label, max_sim, self.label_to_name.get(best_label, "未知")
    
    def _weighted_voting(self, similarities, indices):
        """
        Perform weighted voting with class imbalance handling.
        
        Args:
            similarities: Tensor of shape [K] containing similarity scores
            indices: Tensor of shape [K] containing gallery indices
            
        Returns:
            vote_scores: Dict mapping label_id -> aggregated score
        """
        vote_scores = defaultdict(float)
        
        for val, idx in zip(similarities, indices):
            label = self.gallery_labels[idx].item()
            similarity = val.item()

            # Handle class imbalance: normalize by log of class size
            class_size = self.class_sample_counts.get(label, 1)
            size_weight = 1.0 / np.log(class_size + 1)
            
            # Aggregate score
            vote_scores[label] += similarity * size_weight
        
        return vote_scores
    
    def run_eval(self, test_file, threshold=0.6, margin_threshold=0.1, top_k=5, batch_size=32):
        """
        Run evaluation on test set.
        
        Args:
            test_file: Path to test file (.txt). Format: image_path label_id class_name
            threshold: Absolute similarity threshold
            margin_threshold: Margin threshold between top-1 and top-2
            top_k: Number of nearest neighbors
            
        Returns:
            results: List of dicts containing evaluation results
        """
        test_data = load_test_file(test_file)
        
        if len(test_data) == 0:
            print("测试文件为空", level="error")
            return []
        
        predictions = []
        confidences = []

        image_paths = [item[0] for item in test_data]
        true_names = [item[2] for item in test_data]

        num_batches = (len(image_paths) + batch_size - 1) // batch_size
        with tqdm(total=len(image_paths), desc="测试进度") as pbar:
            for batch_idx in range(num_batches):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, len(image_paths))
                batch_paths = image_paths[start_idx:end_idx]

                batch_results = self._predict_batch(
                    batch_paths,
                    top_k=top_k,
                    threshold=threshold,
                    margin_threshold=margin_threshold
                )

                for pred_label, conf in batch_results:
                    predictions.append(pred_label)
                    confidences.append(conf)

                pbar.update(len(batch_paths))
        
        pred_names = []
        for pred_label in predictions:
            if pred_label == -1:
                pred_names.append(None)
            else:
                pred_names.append(self.label_to_name.get(pred_label, f"Class_{pred_label}"))

        pred_for_metrics = pred_names
        (overall_accuracy, overall_correct, overall_wrong, overall_total,
         per_class_accuracy, per_class_counts, confusion_data) = calculate_metrics_with_unknown(
            pred_for_metrics, true_names
        )
        
        # Print results
        print(f"\n{'='*60}", level="info")
        print("Evaluation Results", level="info")
        print(f"{'='*60}", level="info")
        print(f"Overall Accuracy: {overall_accuracy:.4f} ({overall_accuracy*100:.2f}%)", level="info")
        print(f"Overall: Correct={overall_correct}, Wrong={overall_wrong}, Total={overall_total}", level="info")
        print(f"\nPer-Class Accuracy:", level="info")
        
        classes_dict = {name: name for name in true_names}
        for class_name in sorted(per_class_accuracy.keys()):
            acc = per_class_accuracy[class_name]
            correct_count, wrong_count, total_count = per_class_counts[class_name]
            print(f"  Label {class_name}: "
                  f"Accuracy={acc*100:.2f}%, "
                  f"({correct_count} / {total_count})", level="info")
        print(f"{'='*60}\n", level="info")

        self.last_metrics = {
            "overall_accuracy": overall_accuracy,
            "overall_correct": overall_correct,
            "overall_wrong": overall_wrong,
            "overall_total": overall_total,
            "per_class_accuracy": per_class_accuracy,
            "per_class_counts": per_class_counts,
            "confusion_data": confusion_data,
            "classes_dict": classes_dict
        }
        
        results = []
        for i, (img_path, true_label, true_name) in enumerate(test_data):
            pred_label = predictions[i]
            conf = confidences[i]
            pred_name = pred_names[i] if pred_label != -1 else "Unknown/Rejected"

            results.append({
                "path": img_path,
                "true_label": true_name,
                "true_name": true_name,
                "pred_label": pred_name if pred_label != -1 else None,
                "pred_name": pred_name,
                "confidence": conf,
                "is_correct": (pred_name == true_name) if pred_label != -1 else False,
                "is_unknown": (pred_label == -1)
            })

        return results


def calculate_metrics_with_unknown(predictions, ground_truth):
    assert len(predictions) == len(ground_truth)
    total = sum(1 for g in ground_truth if g is not None)
    correct = sum(1 for p, g in zip(predictions, ground_truth) if p is not None and p == g)
    wrong = total - correct
    overall_accuracy = correct / total if total > 0 else 0.0

    per_class_correct = defaultdict(int)
    per_class_total = defaultdict(int)
    confusion_data = defaultdict(int)

    for pred, true_label in zip(predictions, ground_truth):
        if true_label is None:
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


def main():
    parser = argparse.ArgumentParser(
        description="Open-set recognition evaluation using offline feature gallery",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to model checkpoint (.pt file)")
    parser.add_argument("--template_path", type=str, default=None,
                        help="Path to template file or directory for building gallery")
    parser.add_argument("--test_file", type=str, required=True,
                        help="Path to test file (.txt). Format: image_path label_id class_name")
    parser.add_argument("--output_dir", type=str, default="eval_output",
                        help="Output directory for results")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"],
                        help="Device to run inference on")
    parser.add_argument("--top_k", type=int, default=5,
                        help="Number of nearest neighbors for voting")
    parser.add_argument("--threshold", type=float, default=0.6,
                        help="Absolute similarity threshold for open-set rejection")
    parser.add_argument("--margin_threshold", type=float, default=0.1,
                        help="Margin threshold between top-1 and top-2 scores")
    parser.add_argument("--outlier_threshold", type=float, default=2.0,
                        help="Outlier removal threshold (in standard deviations)")
    parser.add_argument("--batch_size", "-b", type=int, default=32,
                        help="Batch size for inference")
    parser.add_argument("--save_vis", action="store_true", default=False, help="Save visualization images")

    args = parser.parse_args()
    
    template_path = args.template_path
    if template_path is None:
        raise ValueError("Template path is required. Use --template_path.")

    output_dir = args.output_dir + "_eval" + os.path.basename(os.path.dirname(args.model_path))
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize evaluator
    evaluator = OpenSetEvaluator(args.model_path, device=args.device)
    
    # Build gallery
    evaluator.build_gallery_cached(
        template_path,
        outlier_threshold=args.outlier_threshold
    )
    
    # Run evaluation
    results = evaluator.run_eval(
        args.test_file,
        threshold=args.threshold,
        margin_threshold=args.margin_threshold,
        top_k=args.top_k,
        batch_size=args.batch_size
    )

    if evaluator.last_metrics is None:
        print("无有效评估结果，退出。", level="error")
        return
    
    # Save results
    results_file = os.path.join(output_dir, "evaluation_results.txt")
    _write_evaluation_results(results_file, evaluator.last_metrics)

    print(f"结果已保存至: {results_file}", level="info")

    _save_visualizations(output_dir, results, evaluator.last_metrics, args.save_vis)



if __name__ == '__main__':
    main()
