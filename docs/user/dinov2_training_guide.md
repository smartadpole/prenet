# DINOv2 ArcFace 训练指南

## 概述

本指南介绍如何使用 `train_dinov2_arcface_small.py` 训练 DINOv2 基础的 ArcFace 分类模型。该训练系统特别针对**细粒度视觉分类任务**进行了优化，适用于需要区分细微纹理和颜色差异的场景（如肉类分类、食品识别等）。

## 快速开始

### 基本训练命令

```bash
python train_dinov2_arcface_small.py \
    --data_root /path/to/dataset \
    --num_classes 10
```

### 完整训练示例

```bash
python train_dinov2_arcface_small.py \
    --data_root /path/to/dataset \
    --num_classes 10 \
    --train_path /path/to/train.txt \
    --val_path /path/to/val.txt \
    --img_size 224 \
    --batch_size 32 \
    --epochs 30 \
    --output_dir ./output
```

## 数据集准备

### 支持的格式

训练脚本支持两种数据集格式：

#### 格式 1：ImageFolder（目录结构）

```
dataset/
├── train/
│   ├── class_001/
│   │   ├── img1.jpg
│   │   └── img2.jpg
│   └── class_002/
│       ├── img1.jpg
│       └── img2.jpg
└── val/
    ├── class_001/
    └── class_002/
```

**使用方式**：
```bash
python train_dinov2_arcface_small.py \
    --data_root /path/to/dataset \
    --num_classes 10
```

#### 格式 2：文本列表（Text List）

```
dataset/
├── train.txt
└── val.txt
```

**train.txt 格式**（空格或逗号分隔）：
```
class_001/img1.jpg 0
class_001/img2.jpg 0
class_002/img1.jpg 1
class_002/img2.jpg 1
```

**使用方式**：
```bash
python train_dinov2_arcface_small.py \
    --data_root /path/to/dataset \
    --num_classes 10
```

或者显式指定路径：
```bash
python train_dinov2_arcface_small.py \
    --data_root /path/to/dataset \
    --num_classes 10 \
    --train_path /path/to/train.txt \
    --val_path /path/to/val.txt
```

## 参数说明

### 必需参数

| 参数 | 说明 | 示例 |
|------|------|------|
| `--data_root` | 数据集根目录 | `/path/to/dataset` |
| `--num_classes` | 分类类别数量 | `10` |

### 数据集参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--train_path` | 自动检测 | 训练数据路径（文件或目录） |
| `--val_path` | 自动检测 | 验证数据路径（文件或目录） |

### 模型参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--backbone` | `dinov2_vitb14` | DINOv2 模型变体 |
| `--img_size` | `128` | 输入图像尺寸（会自动调整为 patch 大小的倍数） |
| `--embed_dim` | `256` | 嵌入向量维度 |

### 训练参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--epochs` | `20` | 总训练轮数 |
| `--batch_size`, `-b` | `64` | 批次大小 |
| `--num_workers` | `6` | 数据加载工作进程数 |
| `--lr_head`, `-lr` | `3e-4` | 分类头学习率 |
| `--lr_backbone` | `1e-5` | 骨干网络学习率 |
| `--weight_decay` | `0.05` | 权重衰减 |

### ArcFace 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--arc_s` | `32.0` | ArcFace 缩放参数 |
| `--arc_m` | `0.30` | ArcFace 边界参数 |

### 两阶段训练参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--stage1_epochs` | `12` | 第一阶段（冻结骨干）的轮数 |
| `--unfreeze_blocks` | `1` | 第二阶段解冻的最后一层块数 |

### 其他参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--device` | `cuda` | 训练设备（`cuda` 或 `cpu`） |
| `--output_dir` | `output_dinov2_arcface_small` | 模型输出目录 |

## 训练流程

### 两阶段训练策略

训练过程分为两个阶段：

**阶段 1（冻结骨干）**：
- 冻结 DINOv2 骨干网络
- 仅训练投影头和分类头
- 默认持续 12 个 epoch
- 学习率：头层 3e-4，骨干层不更新

**阶段 2（微调骨干）**：
- 解冻最后 N 个 Transformer 块（默认 1 个）
- 同时训练解冻的骨干层、投影头和分类头
- 学习率：头层 3e-4，骨干层 1e-5（较低）

### 学习率调度

- **预热阶段**：第一个 epoch 线性预热
- **主调度**：余弦退火（半周期）
- **公式**：`lr(step) = 0.5 * (1 + cos(π * progress))`

### 数据增强策略

训练使用**细粒度分类优化的数据增强策略**，特别适合需要区分细微纹理和颜色的任务：

**关键特性**：
- **颜色保护**：色相偏移极小（0.01），保护颜色关键特征
- **纹理保护**：模糊强度降低（sigma 上限 1.5），保护纹理细节
- **几何变换保守**：去除错切，减少非物理形变
- **上下文保留**：随机裁剪最小比例提升至 0.8，保留更多上下文

详细参数说明请参考：`docs/dev/training_data_augmentation.md`

## 输出文件

### 模型文件

训练完成后，在 `--output_dir` 目录下会生成：

- `best_train_dinov2_arcface_small.pt`：训练集最佳准确率模型
- `best_val_dinov2_arcface_small.pt`：验证集最佳准确率模型

**模型文件包含**：
- 模型权重（embedder 和 head）
- 类别名称映射（如果可用）
- 训练参数（用于复现）
- 最佳准确率和训练轮数

### 日志文件

- `results_test.txt`：训练过程日志
  - 每轮记录：训练准确率、验证准确率、训练损失
  - 格式：`[时间戳] Iteration {epoch} | train_acc = {acc} | val_acc = {acc} | train_loss = {loss}`

## 训练监控

### 控制台输出

训练过程中会实时显示：
- 当前 epoch 和总 epoch 数
- 每个 batch 的损失值
- 每个 epoch 的训练准确率、验证准确率和平均损失

### 示例输出

```
Epoch 1/20: 100%|████████| 100/100 [00:30<00:00, loss=2.345]
Epoch   1 | train_acc = 0.45230 | val_acc = 0.51234 | train_loss = 2.34567
[Stage2] Unfroze last 1 blocks.
Epoch 13/20: 100%|████████| 100/100 [00:35<00:00, loss=0.123]
Epoch  13 | train_acc = 0.92345 | val_acc = 0.90123 | train_loss = 0.12345
```

## 最佳实践

### 针对细粒度分类任务

1. **图像尺寸**：如果硬件允许，使用更大的 `img_size`（如 224 或 448）
   - 纹理分类任务对分辨率敏感
   - 224x224 和 448x448 的效果差异显著

2. **批次大小**：根据 GPU 内存调整
   - 较大批次（64-128）通常效果更好
   - 如果内存不足，可以减小批次但增加梯度累积

3. **学习率**：保持默认值通常效果良好
   - 如果训练不稳定，可以降低 `lr_head` 至 1e-4
   - 如果收敛慢，可以适当提高（但不超过 1e-3）

4. **训练轮数**：根据数据集大小调整
   - 小数据集（< 1000 样本/类）：可能需要更多轮数（30-50）
   - 大数据集（> 10000 样本/类）：20-30 轮通常足够

### 针对不同场景

**场景 1：类别间差异明显**
- 可以使用默认的数据增强参数
- 可以适当增加 `arc_m`（如 0.35-0.40）增强类别分离

**场景 2：类别间差异细微（细粒度分类）**
- 使用当前优化的数据增强策略（已默认应用）
- 保持较小的 `arc_m`（0.30）避免过度分离
- 考虑使用标签平滑（Label Smoothing）

**场景 3：数据量小**
- 增加 `stage1_epochs`（如 15-20）
- 减少 `unfreeze_blocks`（保持为 1）
- 使用较小的学习率

**场景 4：数据量大**
- 可以减少 `stage1_epochs`（如 8-10）
- 可以增加 `unfreeze_blocks`（如 2-3）进行更充分的微调

## 常见问题

### Q1: 图像尺寸会自动调整吗？

**A**: 是的。如果指定的 `img_size` 不是 DINOv2 patch 大小的倍数，会自动调整为最近的倍数。例如，指定 128 可能调整为 126（如果 patch_size=14）。

### Q2: 如何选择 DINOv2 模型变体？

**A**: 
- `dinov2_vits14`：最小，速度最快，适合快速实验
- `dinov2_vitb14`：默认，平衡性能和速度
- `dinov2_vitl14`：更大，性能更好，需要更多内存
- `dinov2_vitg14`：最大，性能最好，需要大量内存

### Q3: 训练时内存不足怎么办？

**A**: 
- 减小 `batch_size`
- 减小 `img_size`
- 使用较小的 DINOv2 模型（如 `dinov2_vits14`）
- 减少 `num_workers`

### Q4: 如何判断训练是否正常？

**A**: 
- 训练损失应该逐渐下降
- 训练准确率应该逐渐上升
- 验证准确率应该跟随训练准确率上升
- 如果验证准确率停滞或下降，可能过拟合

### Q5: 应该使用哪个模型文件？

**A**: 
- 通常使用 `best_val_dinov2_arcface_small.pt`（验证集最佳）
- 如果验证集分布与测试集不同，可以尝试 `best_train_dinov2_arcface_small.pt`
- 建议在测试集上评估两个模型，选择更好的

## 进阶技巧

### 1. 混合精度训练

训练脚本自动使用混合精度（FP16）训练（如果 CUDA 可用），可以：
- 减少内存使用
- 加快训练速度
- 通常不影响最终精度

### 2. 多 GPU 训练

当前版本不支持多 GPU，如需多 GPU 训练，需要：
- 使用 `torch.nn.DataParallel` 或 `torch.nn.parallel.DistributedDataParallel`
- 调整批次大小和学习率

### 3. 继续训练

当前版本不支持从检查点继续训练，如需此功能，需要：
- 修改代码加载检查点
- 恢复优化器和调度器状态

## 相关文档

- **开发文档**：`docs/dev/dinov2_training_architecture.md` - 系统架构和设计细节
- **数据增强**：`docs/dev/training_data_augmentation.md` - 数据增强策略详解
- **数据加载**：`docs/dev/data_loader_design.md` - 数据加载器设计

## 技术支持

如有问题或建议，请参考：
- 项目 README：`README.md`
- 开发文档：`docs/dev/`
- 变更日志：`CHANGELOG.md`
