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
from train_dinov2_arcface_small import DinoV2Embedder, build_val_tfm

# Import from eval.py in the same directory
import eval as eval_module
load_test_file = eval_module.load_test_file
calculate_metrics = eval_module.calculate_metrics


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
        if device == "cuda" and not torch.cuda.is_available():
            self.device = "cpu"
            print("[警告] CUDA 不可用，使用 CPU")
        
        self.embedder, self.img_size = self._load_model(model_path)
        self.transform = build_val_tfm(self.img_size)
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
        print(f"[信息] 正在加载模型: {model_path}")
        checkpoint = torch.load(model_path, map_location=self.device)
        args = checkpoint.get("args", {})
        
        backbone = args.get("backbone", "dinov2_vitb14")
        embed_dim = args.get("embed_dim", 256)
        img_size = args.get("img_size", 128)
        
        embedder = DinoV2Embedder(backbone, embed_dim, train=False)
        embedder.load_state_dict(checkpoint["embedder"])
        embedder.to(self.device).eval()
        
        print(f"[信息] 模型加载完成: backbone={backbone}, embed_dim={embed_dim}, img_size={img_size}")
        return embedder, img_size
    
    @torch.no_grad()
    def build_gallery(self, template_file, outlier_threshold=2.0):
        """
        Build feature gallery from template images.
        
        Args:
            template_file: Path to template file (.txt). Format: image_path label_id class_name
            outlier_threshold: Threshold for outlier removal (in standard deviations)
        """
        print("[信息] 正在构建特征库...")
        template_data = load_test_file(template_file)
        
        if len(template_data) == 0:
            raise ValueError("模板文件为空，无法构建特征库")
        
        # Extract features for all templates
        features_by_class = defaultdict(list)
        labels_by_class = defaultdict(list)
        
        for img_path, label_id, class_name in tqdm(template_data, desc="提取模板特征"):
            if label_id not in self.label_to_name:
                self.label_to_name[label_id] = class_name
            
            try:
                if not os.path.exists(img_path):
                    print(f"[警告] 图片不存在，跳过: {img_path}")
                    continue
                
                img = Image.open(img_path).convert('RGB')
                img_tensor = self.transform(img).unsqueeze(0).to(self.device)
                feat = self.embedder(img_tensor)  # Already L2 normalized
                features_by_class[label_id].append(feat.cpu())
                labels_by_class[label_id].append(label_id)
                self.class_sample_counts[label_id] += 1
            except Exception as e:
                print(f"[警告] 处理图片失败 {img_path}: {e}")
                continue
        
        # Remove outliers and build gallery
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
        
        print(f"[信息] 特征库构建完成，共 {len(self.gallery_labels)} 个样本，{len(self.label_to_name)} 个类别")
        for label_id, count in sorted(self.class_sample_counts.items()):
            class_name = self.label_to_name[label_id]
            print(f"  类别 {label_id} ({class_name}): {count} 个样本")
    
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
            print(f"[警告] 加载图片失败 {img_path}: {e}")
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
        
        for rank, (val, idx) in enumerate(zip(similarities, indices)):
            label = self.gallery_labels[idx].item()
            similarity = val.item()
            
            # Weight by similarity and rank (inverse rank weighting)
            rank_weight = 1.0 / (rank + 1)
            
            # Handle class imbalance: normalize by log of class size
            class_size = self.class_sample_counts.get(label, 1)
            size_weight = 1.0 / np.log(class_size + 1)
            
            # Aggregate score
            vote_scores[label] += similarity * rank_weight * size_weight
        
        return vote_scores
    
    def run_eval(self, test_file, threshold=0.6, margin_threshold=0.1, top_k=5):
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
        print(f"\n[信息] 开始评估，阈值={threshold}, Top-K={top_k}")
        test_data = load_test_file(test_file)
        
        if len(test_data) == 0:
            print("[错误] 测试文件为空")
            return []
        
        predictions = []
        confidences = []
        true_labels = []
        
        for img_path, true_label, true_name in tqdm(test_data, desc="测试进度"):
            pred_label, conf, pred_name = self.identify(
                img_path, 
                top_k=top_k, 
                threshold=threshold,
                margin_threshold=margin_threshold
            )
            
            predictions.append(pred_label)
            confidences.append(conf)
            true_labels.append(true_label)
        
        # Calculate metrics (treat -1 as None for compatibility)
        pred_for_metrics = [p if p != -1 else None for p in predictions]
        
        (overall_accuracy, overall_correct, overall_wrong, overall_total,
         per_class_accuracy, per_class_counts, confusion_data) = calculate_metrics(
            pred_for_metrics, true_labels
        )
        
        # Count unknown predictions
        unknown_count = sum(1 for p in predictions if p == -1)
        
        # Print results
        print(f"\n{'='*60}")
        print(f"开集识别评估结果")
        print(f"{'='*60}")
        print(f"总体准确率: {overall_accuracy:.4f} ({overall_accuracy*100:.2f}%)")
        print(f"总体: 正确={overall_correct}, 错误={overall_wrong}, 总数={overall_total}")
        print(f"未知类判定: {unknown_count} 个样本被判定为未知类")
        print(f"\n分类别准确率:")
        
        classes_dict = {label_id: name for _, label_id, name in test_data}
        for class_id in sorted(per_class_accuracy.keys()):
            class_name = classes_dict.get(class_id, f"Class_{class_id}")
            acc = per_class_accuracy[class_id]
            correct_count, wrong_count, total_count = per_class_counts[class_id]
            print(f"  类别 {class_id} ({class_name}): "
                  f"准确率={acc*100:.2f}%, "
                  f"({correct_count} / {total_count})")
        print(f"{'='*60}\n")
        
        # Build results list
        results = []
        for i, (img_path, true_label, true_name) in enumerate(test_data):
            pred_label = predictions[i]
            conf = confidences[i]
            pred_name = self.label_to_name.get(pred_label, "未知") if pred_label != -1 else "未知"
            
            results.append({
                "path": img_path,
                "true_label": true_label,
                "true_name": true_name,
                "pred_label": pred_label,
                "pred_name": pred_name,
                "confidence": conf,
                "is_correct": (pred_label == true_label) if pred_label != -1 else False,
                "is_unknown": (pred_label == -1)
            })
        
        return results


def main():
    parser = argparse.ArgumentParser(
        description="Open-set recognition evaluation using offline feature gallery",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to model checkpoint (.pt file)")
    parser.add_argument("--template_file", type=str, required=True,
                        help="Path to template file for building gallery (.txt). Format: image_path label_id class_name")
    parser.add_argument("--test_file", type=str, required=True,
                        help="Path to test file (.txt). Format: image_path label_id class_name")
    parser.add_argument("--output_dir", type=str, default="eval_open_output",
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
    
    args = parser.parse_args()
    
    # Setup output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize evaluator
    evaluator = OpenSetEvaluator(args.model_path, device=args.device)
    
    # Build gallery
    evaluator.build_gallery(args.template_file, outlier_threshold=args.outlier_threshold)
    
    # Run evaluation
    results = evaluator.run_eval(
        args.test_file,
        threshold=args.threshold,
        margin_threshold=args.margin_threshold,
        top_k=args.top_k
    )
    
    # Save results
    results_file = os.path.join(args.output_dir, "open_set_results.txt")
    with open(results_file, 'w', encoding='utf-8') as f:
        f.write("Open-Set Recognition Results\n")
        f.write("="*60 + "\n")
        f.write(f"Template File: {args.template_file}\n")
        f.write(f"Test File: {args.test_file}\n")
        f.write(f"Threshold: {args.threshold}\n")
        f.write(f"Top-K: {args.top_k}\n")
        f.write(f"Margin Threshold: {args.margin_threshold}\n")
        f.write("\n" + "="*60 + "\n\n")
        
        for result in results:
            f.write(f"Image: {result['path']}\n")
            f.write(f"  True: {result['true_name']} (ID: {result['true_label']})\n")
            f.write(f"  Pred: {result['pred_name']} (ID: {result['pred_label']})\n")
            f.write(f"  Confidence: {result['confidence']:.4f}\n")
            f.write(f"  Correct: {result['is_correct']}\n")
            f.write(f"  Unknown: {result['is_unknown']}\n")
            f.write("\n")
    
    print(f"[信息] 结果已保存至: {results_file}")
    print("[信息] 评估完成！")


if __name__ == '__main__':
    main()
