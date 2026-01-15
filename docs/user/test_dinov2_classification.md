# DINOv2 分类测试工具使用指南

## 概述

`test_dinov2_classification.py` 是一个用于测试 DINOv2 ArcFace 模型图像分类功能的工具。该工具可以对指定目录中的图像进行批量分类，并生成分类结果的可视化图表和文本报告。

## 功能特性

- 支持批量处理嵌套目录中的图像文件
- 自动加载训练好的模型检查点
- 生成按类别分组的可视化图表（每个类别单独一张图）
- 导出 CSV 格式的分类结果文件
- 支持 GPU 和 CPU 推理

## 安装要求

确保已安装以下依赖：

```bash
pip install torch torchvision pillow matplotlib tqdm
```

## 使用方法

### 基本用法

```bash
python tools/test_dinov2_classification.py \
    --model_path <模型检查点路径> \
    --image_dir <图像目录路径>
```

### 完整参数说明

| 参数 | 类型 | 必需 | 默认值 | 说明 |
|------|------|------|--------|------|
| `--model_path` | str | 是 | - | 模型检查点文件路径（.pt 文件） |
| `--image_dir` | str | 是 | - | 包含图像的目录路径（支持嵌套目录） |
| `--output_dir` | str | 否 | `test_output` | 结果输出目录 |
| `--device` | str | 否 | `cuda` | 推理设备（`cuda` 或 `cpu`） |
| `--batch_size` | int | 否 | 32 | 批处理大小，用于批量推理加速 |
| `--max_images_per_class` | int | 否 | 20 | 每个类别在可视化中显示的最大图像数量 |
| `--num_classes` | int | 否 | None | 类别数量（如果检查点中未包含） |

### 使用示例

#### 示例 1：基本分类测试

```bash
python tools/test_dinov2_classification.py \
    --model_path checkpoints/best_val_dinov2_arcface_small.pt \
    --image_dir data/test_images
```

#### 示例 2：指定输出目录和设备

```bash
python tools/test_dinov2_classification.py \
    --model_path checkpoints/best_val_dinov2_arcface_small.pt \
    --image_dir data/test_images \
    --output_dir results/my_test \
    --device cpu
```

#### 示例 3：限制可视化图像数量

```bash
python tools/test_dinov2_classification.py \
    --model_path checkpoints/best_val_dinov2_arcface_small.pt \
    --image_dir data/test_images \
    --max_images_per_class 10
```

## 输出结果

工具会在输出目录中生成以下文件：

### 1. 分类结果 CSV 文件

**文件路径：** `{output_dir}/{model_name}/classification_results.csv`

**格式：** 标准 CSV 格式，包含表头行：
```
image_path,class_index,class_name,confidence
```

**示例：**
```csv
image_path,class_index,class_name,confidence
data/test_images/class_001/img1.jpg,0,apple,0.9876
data/test_images/class_001/img2.jpg,0,apple,0.9234
data/test_images/class_002/img1.jpg,1,banana,0.8765
```

**说明：** CSV 格式便于在 Excel、Python pandas 等工具中打开和分析。

### 2. 可视化图表

**文件路径：** `{output_dir}/{model_name}/classification_visualization_class_{class_idx}_{class_name}.png`

每个类别会生成一张独立的可视化图表，包含：
- 类别标题（类别索引、类别名称、图像数量）
- 该类别下的所有图像（按置信度降序排列）
- 每个图像的置信度分数

图表布局：4 列网格，自动计算行数。

## 注意事项

1. **模型检查点格式**：工具期望检查点文件包含以下键：
   - `embedder`：嵌入器模型状态字典
   - `head`：分类头模型状态字典
   - `args`：模型参数（包含 `embed_dim`、`num_classes`、`backbone`、`img_size`、`arc_s`、`arc_m`）
   - `classes`：类别名称列表（可选）

2. **图像格式**：支持 PIL 可以打开的所有图像格式（JPEG、PNG、BMP 等）

3. **内存使用**：处理大量图像时，注意内存使用情况。如果遇到内存不足，可以分批处理图像目录。

4. **CUDA 可用性**：如果指定 `--device cuda` 但 CUDA 不可用，工具会自动回退到 CPU。

## 故障排除

### 问题：找不到类别数量

**错误信息：** `Cannot determine num_classes from checkpoint`

**解决方案：** 使用 `--num_classes` 参数显式指定类别数量：
```bash
python tools/test_dinov2_classification.py \
    --model_path checkpoints/model.pt \
    --image_dir data/test_images \
    --num_classes 100
```

### 问题：图像加载失败

**现象：** 控制台输出 `[Warning] Failed to process ...`

**可能原因：**
- 图像文件损坏
- 不支持的图像格式
- 文件路径错误

**解决方案：** 检查图像文件是否完整，确保使用标准图像格式。

### 问题：CUDA 内存不足

**解决方案：** 
- 使用 `--device cpu` 切换到 CPU 模式
- 或者减少处理的图像数量（分批处理）

## 相关文件

- 模型训练脚本：`train_dinov2_arcface_small.py`
- 数据加载工具：`data/unified_data_loader.py`
- 工具函数：`utils/file.py`（`walk_image` 函数）

