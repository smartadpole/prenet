# 开集识别评估工具使用指南

## 概述

`eval_open.py` 是一个用于开集识别评估的工具，支持基于离线特征库的度量学习识别方案。该工具可以将模型从分类器任务转变为度量学习任务，实现"即插即用"的识别，无需重新训练即可添加新类别。

## 功能特性

- 支持从模板图构建离线特征库
- 使用 Top-K 最近邻检索进行识别
- 加权投票机制处理类别模板不均衡问题
- 双阈值判定逻辑识别未知类
- 支持批量评估和详细结果统计

## 安装要求

确保已安装以下依赖：

```bash
pip install torch torchvision pillow numpy tqdm
```

## 核心概念

### 开集识别 vs 闭集识别

- **闭集识别**：只能识别训练时见过的类别，类别数量固定
- **开集识别**：可以识别训练时未见过的类别，通过相似度判定是否为"未知类"

### 特征库（Gallery）

特征库是预先计算的模板图特征集合，包含：
- 各类别模板图的特征向量
- 对应的类别标签和名称
- 经过离群点清洗的纯净特征

### 识别流程

1. 提取查询图像的特征向量
2. 计算与特征库中所有样本的相似度
3. 获取 Top-K 最近邻
4. 使用加权投票机制聚合类别得分
5. 应用双阈值判定是否为已知类或未知类

## 使用方法

### 基本用法

```bash
python tools/eval_open.py \
    --model_path <模型检查点路径> \
    --template_path <模板路径> \
    --test_file <测试文件路径>
```

### 完整参数说明

| 参数 | 类型 | 必需 | 默认值 | 说明 |
|------|------|------|--------|------|
| `--model_path` | str | 是 | - | 模型检查点文件路径（.pt 文件） |
| `--template_path` | str | 是 | - | 模板文件或目录路径（用于构建特征库） |
| `--test_file` | str | 是 | - | 测试文件路径 |
| `--output_dir` | str | 否 | `eval_output` | 结果输出目录 |
| `--device` | str | 否 | `cuda` | 推理设备（`cuda` 或 `cpu`） |
| `--top_k` | int | 否 | 5 | Top-K 最近邻数量 |
| `--threshold` | float | 否 | 0.6 | 绝对相似度阈值（用于判定未知类） |
| `--margin_threshold` | float | 否 | 0.1 | 相对增益阈值（Top-1 与 Top-2 的差值） |
| `--outlier_threshold` | float | 否 | 2.0 | 离群点判定阈值（倍数标准差） |
| `--batch_size` | int | 否 | 32 | 批量推理大小 |
| `--save_vis` | flag | 否 | - | 保存可视化结果（默认关闭） |

### 文件格式

#### 模板文件格式

模板文件用于构建特征库，每行包含三列（空格或制表符分隔）：

```
<绝对图片路径> <标签ID> <类别名称>
```

示例：
```
/path/to/template1.jpg 0 类别A
/path/to/template2.jpg 0 类别A
/path/to/template3.jpg 1 类别B
/path/to/template4.jpg 1 类别B
```

#### 模板目录格式

```
<template_root>/
├── <类别名称1>/
│   ├── img1.jpg
│   └── ...
└── <类别名称2>/
    └── ...
```

#### 测试文件格式

测试文件包含待评估的图像，格式与模板文件相同：

```
<绝对图片路径> <标签ID> <类别名称>
```

示例：
```
/path/to/test1.jpg 0 类别A
/path/to/test2.jpg 1 类别B
/path/to/test3.jpg 2 未知类别
```

## 使用示例

### 示例 1：基本开集识别评估

```bash
python tools/eval_open.py \
    --model_path output/best_val_model.pt \
    --template_path data/templates.txt \
    --test_file data/test_open_set.txt
```

### 示例 2：自定义阈值参数

```bash
python tools/eval_open.py \
    --model_path output/best_val_model.pt \
    --template_path data/templates.txt \
    --test_file data/test_open_set.txt \
    --threshold 0.65 \
    --margin_threshold 0.15 \
    --top_k 10
```

### 示例 3：使用 CPU 推理

```bash
python tools/eval_open.py \
    --model_path output/best_val_model.pt \
    --template_path data/templates.txt \
    --test_file data/test_open_set.txt \
    --device cpu
```

### 示例 4：自定义输出目录

```bash
python tools/eval_open.py \
    --model_path output/best_val_model.pt \
    --template_path data/templates.txt \
    --test_file data/test_open_set.txt \
    --output_dir my_eval_results
```

## 参数调优指南

### Top-K 参数

- **较小值（3-5）**：计算速度快，但可能忽略重要信息
- **较大值（10-20）**：考虑更多近邻，但计算开销增加
- **推荐值**：5-10，平衡准确率和效率

### 绝对阈值（threshold）

- **较高值（0.7-0.8）**：更严格，更多样本被判定为未知类
- **较低值（0.5-0.6）**：更宽松，可能误判未知类为已知类
- **推荐值**：0.6-0.7，根据实际数据分布调整

### 相对阈值（margin_threshold）

- **较高值（0.15-0.2）**：要求类别间区分度更大
- **较低值（0.05-0.1）**：允许类别间相似度较高
- **推荐值**：0.1-0.15，避免模糊样本被错误分类

### 离群点阈值（outlier_threshold）

- **较高值（2.5-3.0）**：保留更多模板样本
- **较低值（1.5-2.0）**：更严格过滤异常样本
- **推荐值**：2.0，基于统计学的 2 倍标准差原则

## 输出结果

### 控制台输出

工具运行时会显示：
- 模型加载信息
- 特征库构建进度和统计信息
- 测试评估进度
- 总体和分类别准确率统计

示例输出：
```
[信息] 正在加载模型: output/best_val_model.pt
[信息] 模型加载完成: backbone=dinov2_vitb14, embed_dim=256, img_size=128
[信息] 正在构建特征库...
提取模板特征: 100%|████████████| 100/100 [00:05<00:00, 18.23it/s]
[信息] 特征库构建完成，共 100 个样本，10 个类别
  类别 0 (类别A): 15 个样本
  类别 1 (类别B): 12 个样本
  ...

[信息] 开始评估，阈值=0.6, Top-K=5
测试进度: 100%|████████████| 200/200 [00:10<00:00, 19.45it/s]

============================================================
Evaluation Results
============================================================
Overall Accuracy: 0.8750 (87.50%)
Overall: Correct=175, Wrong=25, Total=200
Per-Class Accuracy:
  Label 类别A: Accuracy=90.00%, (18 / 20)
  Label 类别B: Accuracy=85.00%, (17 / 20)
  ...
============================================================
```

### 结果文件

结果保存在 `--output_dir` 指定的目录中，文件名为 `evaluation_results.txt`，包含：
- 总体准确率、正确数、错误数、总数
- 分类别准确率统计
- 未知预测计入错误统计

此外可通过 `--save_vis` 生成按类别可视化图像：
- Top 10 正确样本（按置信度排序）
- 所有错误样本（包含真实/预测标签）
- 未知预测标记为 `Unknown/Rejected`

模板库默认启用缓存，缓存目录为模板目录同级的 `gallery_cache_<模型版本号>`；可视化默认关闭。

## 常见问题

### Q1: 如何选择合适的阈值？

**A:** 建议先用默认值（threshold=0.6, margin_threshold=0.1）进行评估，然后根据结果调整：
- 如果未知类样本被误判为已知类，提高 threshold
- 如果已知类样本被误判为未知类，降低 threshold
- 如果类别间混淆严重，提高 margin_threshold

### Q2: 模板文件需要多少样本？

**A:** 建议每个类别至少 3-5 个模板样本，样本越多特征库越稳定。但也要注意：
- 样本过多会增加计算开销
- 建议使用代表性强的样本（不同视角、光照条件等）
- 工具会自动进行离群点清洗，过滤异常样本

### Q3: 如何处理类别模板不均衡问题？

**A:** 工具内置了加权投票机制，使用类别大小归一化（`1/log(N_c + 1)`）处理不均衡问题。如果仍有问题，可以：
- 增加样本量少的类别的模板数量
- 调整 `top_k` 参数，让更多近邻参与投票
- 检查模板质量，确保样本具有代表性

### Q4: 特征库构建很慢怎么办？

**A:** 特征库构建是一次性操作，构建完成后可以重复使用。如果模板数量很大（>1000），可以考虑：
- 使用 GPU 加速（`--device cuda`）
- 对样本过多的类别进行聚类下采样
- 使用更具代表性的样本，减少模板数量

### Q5: 如何添加新类别？

**A:** 开集识别的优势就是可以动态添加类别：
1. 准备新类别的模板图
2. 将模板图路径添加到模板文件中
3. 重新运行 `build_gallery()` 构建特征库
4. 无需重新训练模型

## 性能优化建议

1. **使用 GPU**：如果可用，使用 `--device cuda` 显著加速特征提取
2. **批量处理**：工具内部已实现批量处理，无需手动分批
3. **特征库缓存**：特征库构建完成后可以保存，避免重复计算（未来版本将支持）
4. **索引优化**：当特征库样本数量很大（>10000）时，考虑使用 FAISS 等向量索引库（未来版本将支持）

## 相关文档

- **技术文档**：`docs/dev/eval_open_set.md` - 详细的技术实现文档
- **闭集评估**：`tools/eval.py` - 闭集分类评估工具
- **训练指南**：`docs/user/dinov2_training_guide.md` - DINOv2 模型训练指南
