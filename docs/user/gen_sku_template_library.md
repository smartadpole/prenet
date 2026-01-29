# 模板库生成工具使用指南

## 概述

`gen_sku.py` 是一个基于 fastdup 聚类的模板库生成工具，专门用于开集识别场景。该工具通过聚类算法从大量图片中智能采样，生成具有代表性的模板库，确保在有限的模板数量（10-20 张）内覆盖尽可能多的场景变化（光线、烟雾、角度等）。

## 功能特性

- **聚类中心采样**：从每个聚类簇中提取距离中心最近的图片，确保代表性
- **混合采样策略**：结合中心样本和边缘样本，增强对极端环境的包容度
- **灵活输入方式**：支持（1）根目录递归扫描，识别多层子目录下“叶子”类别目录；（2）文件列表（路径、ID、类别名，格式同 `eval.load_test_file`）
- **复杂环境优化**：针对光线、烟雾、角度变化等复杂场景优化参数

## 安装要求

确保已安装以下依赖：

```bash
pip install fastdup pandas
```

## 使用方法

### 基本用法

**根据 `--input` 类型自动选择模式：**

- **输入为文件**：文件列表模式。文件每行三列（制表符/空格/逗号分隔）：`图片绝对路径`、`ID`、`类别名`，格式与 `eval.load_test_file` 一致。对列表中涉及的所有目录做一次 fastdup 特征分析，再按标签组内做聚类采样（中心或混合）。
- **输入为目录**：递归目录模式。递归扫描根目录下所有子目录，将**包含图片且不为其它含图目录父目录**的“叶子”目录视为类别，对每个类别分别做聚类采样。

#### 递归目录模式（多层嵌套目录）

```bash
python tools/gen_sku.py \
    --input ./my_dataset \
    --method center \
    --num_templates 15
```

#### 文件列表模式

```bash
python tools/gen_sku.py \
    --input /path/to/list.txt \
    --method hybrid \
    --num_templates 20 \
    --edge_ratio 0.2
```

列表文件示例（`list.txt`）：
```
/path/to/img1.jpg	0	类别A
/path/to/img2.png	0	类别A
/path/to/img3.jpg	1	类别B
```

### 参数说明

| 参数 | 类型 | 必需 | 默认值 | 说明 |
|------|------|------|--------|------|
| `--input` | str | 是 | - | 数据集根目录（递归扫描叶子类别）或文件列表路径（三列：路径、ID、类别名） |
| `--output` | str | 否 | 见下 | 输出模板列表 txt 文件路径。目录输入时默认 `<input>_templates.txt`；文件输入时默认与输入同路径 |
| `--method` | str | 否 | `center` | 采样方法：`center`（中心采样）或 `hybrid`（混合采样） |
| `--num_templates` | int | 否 | 20 | 每个类别生成的模板数量 |
| `--edge_ratio` | float | 否 | 0.2 | 边缘样本比例（仅混合模式，0.0-1.0） |
| `--num_em_iter` | int | 否 | 30 | KMeans 迭代次数（针对复杂环境优化） |

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
    --input ./my_dataset \
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
    --input ./my_dataset \
    --method hybrid \
    --num_templates 20 \
    --edge_ratio 0.2
```

### 输入模式说明

- **目录输入**：递归扫描，将“叶子”目录（包含图片且无子目录再含图）视为类别。支持多层嵌套，如 `my_dataset/brand/model/sku/` 中 `sku` 为叶子类别。
- **文件列表输入**：三列（路径、ID、类别名），对列表中涉及的所有目录做一次 fastdup 分析，再按标签组内做聚类采样。

## 使用示例

### 示例 1：递归目录 + 中心采样

```bash
# 递归扫描 dataset，叶子目录为类别，每个类别 15 张模板
python tools/gen_sku.py \
    --input ./dataset \
    --method center \
    --num_templates 15 \
    --output ./template_library/templates.txt
```

### 示例 2：递归目录 + 混合采样

```bash
# 递归目录，混合采样增强对极端情况的包容度
python tools/gen_sku.py \
    --input ./dataset \
    --method hybrid \
    --num_templates 20 \
    --edge_ratio 0.2 \
    --num_em_iter 30
```

### 示例 3：文件列表模式

```bash
# 从 list.txt（路径、ID、类别名）生成模板库
python tools/gen_sku.py \
    --input /path/to/list.txt \
    --method center \
    --num_templates 10 \
    --output ./templates_from_list.txt
```

## 输出说明

### 输出文件

工具输出**单个 txt 文件**（由 `--output` 指定），每行一条模板记录，格式为：

```
<绝对路径>,<label_id>,<类别名>
```

示例：

```
/path/to/img1.jpg,0,类别A
/path/to/img2.png,0,类别A
/path/to/img3.jpg,1,类别B
```

该文件可直接作为开集评估（如 `eval_open.py`）的模板列表使用。

### 临时工作目录

聚类过程会在**输出文件所在目录**下创建临时目录 `work_dirs/`，按类别子目录（如 `work_dirs/label_0/`）存放 fastdup 中间文件。**写出结果后工具会自动清理该临时目录**，无需手动删除。

## 注意事项

1. **样本数量限制**：如果某个类别的图片数量少于所需模板数，工具会自动使用所有图片作为模板。

2. **工作目录**：工具会在输出文件所在目录下创建临时工作目录 `work_dirs/`，包含 fastdup 的中间文件；写出结果后会自动清理，无需手动删除。

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
