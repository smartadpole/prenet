# 模板库生成工具使用指南

## 概述

`gen_sku.py` 是一个基于 fastdup 聚类的模板库生成工具，专门用于开集识别场景。该工具通过聚类算法从大量图片中智能采样，生成具有代表性的模板库，确保在有限的模板数量（10-20 张）内覆盖尽可能多的场景变化（光线、烟雾、角度等）。

## 功能特性

- **聚类中心采样**：从每个聚类簇中提取距离中心最近的图片，确保代表性
- **混合采样策略**：结合中心样本和边缘样本，增强对极端环境的包容度
- **批量处理支持**：自动处理多个类别文件夹
- **智能模式检测**：自动识别批量模式或单类别模式
- **复杂环境优化**：针对光线、烟雾、角度变化等复杂场景优化参数

## 安装要求

确保已安装以下依赖：

```bash
pip install fastdup pandas
```

## 使用方法

### 基本用法

#### 批量处理多个类别

```bash
python tools/gen_sku.py \
    --input_dir ./my_dataset \
    --batch \
    --method center \
    --num_templates 15
```

#### 处理单个类别

```bash
python tools/gen_sku.py \
    --input_dir ./my_dataset/category1 \
    --single \
    --method hybrid \
    --num_templates 20
```

### 参数说明

| 参数 | 类型 | 必需 | 默认值 | 说明 |
|------|------|------|--------|------|
| `--input_dir` | str | 是 | - | 输入目录（类别文件夹或根目录） |
| `--output_dir` | str | 否 | `template_library` | 模板库输出目录 |
| `--method` | str | 否 | `center` | 采样方法：`center`（中心采样）或 `hybrid`（混合采样） |
| `--num_templates` | int | 否 | 15 | 每个类别生成的模板数量 |
| `--edge_ratio` | float | 否 | 0.2 | 边缘样本比例（仅混合模式，0.0-1.0） |
| `--num_em_iter` | int | 否 | 30 | KMeans 迭代次数（针对复杂环境优化） |
| `--batch` | flag | 否 | - | 强制批量处理模式 |
| `--single` | flag | 否 | - | 强制单类别处理模式 |

### 采样方法详解

#### 1. 中心采样（Center Sampling）

**原理：** 将图片聚类为 K 个簇（K = 模板数量），从每个簇中提取距离中心最近的图片。

**适用场景：**
- 需要确保模板覆盖不同场景/角度
- 数据分布相对均匀
- 追求稳定性和代表性

**示例：**
```bash
python tools/gen_sku.py \
    --input_dir ./my_dataset \
    --method center \
    --num_templates 15
```

#### 2. 混合采样（Hybrid Sampling）

**原理：** 80% 中心样本（代表性）+ 20% 边缘样本（极端情况）

**适用场景：**
- 环境复杂（烟雾、强光、极端角度）
- 需要增强开集识别的包容度
- 希望同时兼顾代表性和鲁棒性

**示例：**
```bash
python tools/gen_sku.py \
    --input_dir ./my_dataset \
    --method hybrid \
    --num_templates 20 \
    --edge_ratio 0.2
```

### 处理模式

#### 自动检测模式（推荐）

如果不指定 `--batch` 或 `--single`，工具会自动检测：

- 如果输入目录包含子目录 → 批量模式
- 如果输入目录直接包含图片 → 单类别模式

```bash
# 自动检测为批量模式
python tools/gen_sku.py --input_dir ./my_dataset --num_templates 15

# 自动检测为单类别模式
python tools/gen_sku.py --input_dir ./my_dataset/category1 --num_templates 15
```

#### 批量处理模式

处理输入目录下的所有类别子目录：

```bash
python tools/gen_sku.py \
    --input_dir ./my_dataset \
    --batch \
    --num_templates 15
```

**目录结构示例：**
```
my_dataset/
├── category1/
│   ├── img1.jpg
│   ├── img2.jpg
│   └── ...
├── category2/
│   ├── img1.jpg
│   └── ...
└── category3/
    └── ...
```

#### 单类别处理模式

处理单个类别文件夹：

```bash
python tools/gen_sku.py \
    --input_dir ./my_dataset/category1 \
    --single \
    --num_templates 15
```

## 使用示例

### 示例 1：为开集识别生成模板库

```bash
# 使用中心采样，每个类别生成 15 张模板
python tools/gen_sku.py \
    --input_dir ./dataset \
    --batch \
    --method center \
    --num_templates 15 \
    --output_dir ./template_library
```

### 示例 2：复杂环境下的混合采样

```bash
# 使用混合采样，增加对极端情况的包容度
python tools/gen_sku.py \
    --input_dir ./dataset \
    --batch \
    --method hybrid \
    --num_templates 20 \
    --edge_ratio 0.2 \
    --num_em_iter 30
```

### 示例 3：处理单个类别

```bash
# 为单个类别生成模板库
python tools/gen_sku.py \
    --input_dir ./dataset/category_A \
    --single \
    --method center \
    --num_templates 10 \
    --output_dir ./templates_category_A
```

## 输出说明

### 目录结构

```
template_library/
├── category1/
│   ├── template_c0.jpg
│   ├── template_c1.jpg
│   └── ...
├── category2/
│   ├── template_c0.jpg
│   └── ...
└── work_category1/          # 临时工作目录
    └── kmeans_assignments.csv
```

### 文件命名规则

- **中心采样模式**：`template_c{cluster_id}.{ext}`
  - 例如：`template_c0.jpg`, `template_c1.png`
  
- **混合采样模式**：
  - 中心样本：`center_c{cluster_id}_{original_name}`
  - 边缘样本：`edge_c{cluster_id}_{original_name}`

## 注意事项

1. **样本数量限制**：如果某个类别的图片数量少于所需模板数，工具会自动使用所有图片作为模板。

2. **工作目录**：工具会在输出目录下创建临时工作目录（`work_*`），包含 fastdup 的中间文件。可以手动删除这些目录以节省空间。

3. **性能考虑**：
   - 聚类计算时间与图片数量和迭代次数成正比
   - 建议 `num_em_iter` 设置为 20-30，平衡效果和速度
   - 对于超大数据集，可以考虑先进行初步筛选

4. **环境要求**：
   - 确保 fastdup 已正确安装
   - 需要足够的磁盘空间存储临时文件
   - 建议使用 SSD 以提升 I/O 性能

## 常见问题

**Q: 为什么选择聚类而不是随机采样？**

A: 在光线、烟雾、角度变化剧烈的场景下，聚类能确保在有限的模板数量内覆盖尽可能多的"极端情况"，而随机采样容易抽到相似的图像，导致模板库覆盖不全。

**Q: 中心采样和混合采样如何选择？**

A: 
- 如果数据分布相对均匀，环境变化不剧烈 → 使用中心采样
- 如果环境复杂（烟雾、强光、极端角度）→ 使用混合采样

**Q: 模板数量如何确定？**

A: 一般建议 10-20 张。太少可能覆盖不全，太多可能引入冗余。可以根据实际效果调整。

**Q: 处理失败怎么办？**

A: 检查：
1. 输入目录是否存在且包含图片
2. fastdup 是否正确安装
3. 磁盘空间是否充足
4. 查看错误日志了解具体原因
