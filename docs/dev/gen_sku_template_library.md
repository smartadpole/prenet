# 模板库生成工具技术文档

## 概述

本文档描述 `tools/gen_sku.py` 的技术实现细节，面向开发者和维护者。该工具基于 fastdup 实现聚类采样，用于从大量图片中生成具有代表性的模板库，适用于开集识别场景。

## 架构设计

### 模块结构

```
gen_sku.py
├── 核心采样
│   ├── perform_sampling_from_df()       # 从聚类结果 DataFrame 执行中心/混合采样
│   └── _run_fastdup_kmeans()            # 调用 fastdup.run_kmeans(input_dir=..., work_dir=...)
├── 流程与数据
│   ├── process_by_dataframe()           # 按类别组内独立聚类、写结果列表并在 finally 中清理 work_root
│   ├── process_by_file() / process_by_directory()
│   ├── load_data_from_file()            # 解析文件列表为 DataFrame（与 eval.load_test_file 一致）
│   └── _normalize_kmeans_df() / _pick_assignments_csv()
├── 工具
│   └── mkdir_simple()                   # 从 utils.file 导入
└── main()                               # 参数解析与入口
```

### 数据流

#### 中心采样流程

```
输入目录 → 检查图片数量
    ↓
fastdup.run_kmeans() → 聚类为 K 个簇
    ↓
读取 kmeans_assignments.csv → 聚类结果 DataFrame
    ↓
按 cluster 分组 → 取 distance 最小的样本
    ↓
复制到输出目录 → 模板库
```

#### 混合采样流程

```
输入目录 → 检查图片数量
    ↓
fastdup.run_kmeans() → 聚类为 (总数-边缘数) 个簇
    ↓
读取聚类结果 → 提取中心样本（distance 最小）
    ↓
计算簇的平均距离 → 选择最分散的簇
    ↓
从分散簇中提取边缘样本（distance 最大）
    ↓
合并中心+边缘样本 → 复制到输出目录
```

## 核心功能实现

### 1. 聚类中心采样 (`method2_template_sampling`)

**职责：** 将图片聚类为 K 个簇，从每个簇中提取距离中心最近的图片作为模板

**设计决策：**
- 使用 `fastdup.run_kmeans()` 进行特征提取和聚类
- 通过 `groupby('cluster')['distance'].idxmin()` 找到每个簇的中心样本
- 默认迭代次数 30，针对复杂环境（光线、烟雾）优化

**关键代码逻辑：**
```python
# 运行 KMeans 聚类
ret = fastdup.run_kmeans(
    input_dir=input_dir,
    work_dir=work_dir,
    num_clusters=num_templates,
    num_em_iter=num_em_iter,
    verbose=False
)

# 读取聚类结果
df_clusters = pd.read_csv(kmeans_assignments_file)

# 去重：确保每张图片只属于一个簇（保留距离最小的）
df_clusters = df_clusters.loc[df_clusters.groupby('filename')['distance'].idxmin()]

# 核心采样：每个簇取距离中心最近的样本
df_templates = df_clusters.loc[df_clusters.groupby('cluster')['distance'].idxmin()]
```

**边界情况处理：**
- 图片数量少于模板数 → 直接使用所有图片
- 文件不存在 → 跳过并记录警告
- 聚类失败 → 返回 False 并输出错误信息

### 2. 混合采样 (`method2_hybrid_sampling`)

**职责：** 结合中心样本和边缘样本，增强对极端环境的包容度

**设计决策：**
- 80% 中心样本：每个簇的代表性图片
- 20% 边缘样本：从分布最分散的簇中提取最远的图片
- 通过 `cluster_variance = groupby('cluster')['distance'].mean()` 识别分散簇

**关键代码逻辑：**
```python
# 计算簇的平均距离，识别最分散的簇
cluster_variance = df_clusters.groupby('cluster')['distance'].mean().sort_values(ascending=False)
diverse_clusters = cluster_variance.head(num_edges).index.tolist()

# 从分散簇中提取边缘样本（距离中心最远）
for c_id in diverse_clusters:
    cluster_data = df_clusters[df_clusters['cluster'] == c_id]
    edge_sample = cluster_data.nlargest(1, 'distance')
    edge_list.append(edge_sample)
```

**优势：**
- 中心样本保证代表性
- 边缘样本增强对极端情况的包容度
- 适合复杂环境（烟雾、强光、极端角度）

### 3. 批量处理 (`process_categories`)

**职责：** 遍历根目录下的所有类别文件夹，批量生成模板库

**设计决策：**
- 自动识别类别文件夹（排除文件）
- 为每个类别创建独立的工作目录
- 统计成功/失败数量

**关键代码逻辑：**
```python
category_folders = [f for f in os.listdir(root_data_dir) 
                   if os.path.isdir(os.path.join(root_data_dir, f))]

for category_folder in category_folders:
    full_path = os.path.join(root_data_dir, category_folder)
    work_dir = os.path.join(output_dir, f"work_{category_folder}")
    
    # 调用采样函数
    success = method2_template_sampling(...)
```

### 4. 模式自动检测

**职责：** 根据输入目录结构自动判断处理模式

**设计决策：**
- 检查输入目录是否包含子目录
- 有子目录 → 批量模式
- 无子目录 → 单类别模式

**关键代码逻辑：**
```python
subdirs = [f for f in os.listdir(args.input_dir) 
          if os.path.isdir(os.path.join(args.input_dir, f))]

if len(subdirs) > 0:
    # 批量模式
    process_categories(...)
else:
    # 单类别模式
    method2_template_sampling(...)
```

## 技术细节

### Fastdup 集成

**API 使用：**
- 直接调用 `fastdup.run_kmeans(input_dir=image_paths, work_dir=class_work_dir, num_clusters=..., num_em_iter=..., verbose=False)`，其中 `image_paths` 为当前类别图片路径列表。**必须使用 `input_dir` 传入路径列表**，不得通过 subset.csv + `fd.run(annotations=...)` 再 `run_kmeans(work_dir=...)` 的方式间接调用，否则会导致聚类错误（v0.1.18 修复）。
- 输出文件：`kmeans_assignments.csv`（默认在 `work_dir` 下，可通过 `_pick_assignments_csv(work_dir)` 解析）
- CSV 格式：含 `filename`、`cluster`、`distance` 等列（列名可能为 `cluster_id`/`dist`，由 `_normalize_kmeans_df` 统一为 `cluster`/`distance`）

**参数调优：**
- `num_em_iter=30`: 增加迭代次数以应对光线和烟雾干扰
- `num_clusters`: 等于所需模板数量

### 数据去重策略

**问题：** fastdup 输出中同一张图片可能出现在多条记录（多簇或重复），若直接按簇采样会导致同一图片被多次选中或采样异常。

**解决方案：** 读取 `kmeans_assignments.csv` 并完成列名规范化（`_normalize_kmeans_df`）后，**先按 filename 去重，再执行簇内采样**。去重规则：每张图片只保留一条记录，取该图片在所有簇中 `distance` 最小的那条（即其“最归属”的簇）。
```python
# 按 filename 分组，保留 distance 最小的记录
df_clusters = df_clusters.loc[df_clusters.groupby('filename')['distance'].idxmin()]
```
此后再进行 `groupby('cluster')['distance'].idxmin()` 等簇内采样。

### 文件命名规则

**中心采样：**
- 格式：`template_c{cluster_id}.{ext}`
- 目的：便于识别模板所属的聚类簇

**混合采样：**
- 中心样本：`center_c{cluster_id}_{original_name}`
- 边缘样本：`edge_c{cluster_id}_{original_name}`
- 目的：区分样本类型和所属簇

### 错误处理

**检查点：**
1. 输入目录存在性检查
2. 图片文件数量检查
3. 聚类结果文件存在性检查
4. 文件复制时的存在性检查

**错误处理策略：**
- 文件不存在 → 跳过并记录警告，继续处理
- 聚类失败 → 返回 False，停止处理当前类别
- 样本不足 → 使用所有图片，输出警告

## 性能考虑

### 计算复杂度

- **特征提取**：O(N)，N 为图片数量
- **KMeans 聚类**：O(N × K × I)，K 为簇数，I 为迭代次数
- **采样选择**：O(N log N)，主要是排序操作

### 优化建议

1. **迭代次数**：`num_em_iter` 建议 20-30，平衡效果和速度
2. **批量处理**：对于大量类别，考虑并行处理
3. **磁盘 I/O**：使用 SSD 提升文件复制速度
4. **内存管理**：大数据集时考虑分批处理

## 依赖关系

### 外部依赖

- `fastdup`: 特征提取和聚类
- `pandas`: 数据处理
- `shutil`: 文件复制
- `utils.file.mkdir_simple`: 目录创建工具

### 内部依赖

- 无其他内部模块依赖

## 扩展性

### 未来可能的改进

1. **并行处理**：支持多进程/多线程批量处理
2. **质量筛选**：在聚类前剔除质量极差的图片
3. **自适应聚类数**：根据数据分布自动调整簇数
4. **可视化支持**：生成聚类结果的可视化图表
5. **增量更新**：支持在现有模板库基础上添加新类别

### 接口设计

当前接口设计清晰，易于扩展：
- 采样方法通过 `method` 参数选择
- 可以轻松添加新的采样策略（如 `method3_quality_based`）
- 批量处理逻辑与采样逻辑分离，便于维护

## 测试建议

### 单元测试

1. **采样函数测试**：
   - 测试正常情况（图片数 > 模板数）
   - 测试边界情况（图片数 < 模板数）
   - 测试文件不存在的情况

2. **批量处理测试**：
   - 测试多类别处理
   - 测试单类别处理
   - 测试空目录处理

### 集成测试

1. **端到端测试**：
   - 完整流程测试（输入 → 聚类 → 输出）
   - 不同采样方法对比测试
   - 复杂环境数据测试

2. **性能测试**：
   - 大数据集处理时间测试
   - 内存使用测试
   - 磁盘 I/O 性能测试

## 已知问题与限制

1. **fastdup 依赖**：需要确保 fastdup 正确安装和配置
2. **内存限制**：超大数据集可能需要分批处理
3. **工作目录清理**：`process_by_dataframe` 在写出结果后会在 `finally` 中自动清理临时工作目录（`work_root`），无需手动清理
4. **文件格式支持**：目前支持常见图片格式（jpg, png, bmp, tiff）

## 相关文档

- 用户指南：`docs/user/gen_sku_template_library.md`
- Fastdup 文档：参考 fastdup 官方文档
- 开集识别评估：`docs/user/eval_open_set.md`
