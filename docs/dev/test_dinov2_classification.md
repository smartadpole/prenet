# DINOv2 分类测试工具技术文档

## 概述

本文档描述 `tools/test_dinov2_classification.py` 的技术实现细节，面向开发者和维护者。

## 架构设计

### 模块结构

```
test_dinov2_classification.py
├── 模型加载模块
│   └── load_model()
├── 分类推理模块
│   ├── classify()
│   └── classify_image()
├── 可视化模块
│   └── visualize_by_category()
└── 主程序
    └── main()
```

### 数据流

```
图像目录 → walk_image() → 图像路径列表
    ↓
分批处理 (batch_size) → classify_batch() → 批量分类结果
    ↓
image_results 字典 → visualize_by_category() → PNG 图表
    ↓
results_by_class 字典 → CSV 文件导出
```

## 核心功能实现

### 1. 模型加载 (`load_model`)

**职责：** 从检查点文件加载训练好的模型

**实现细节：**
- 使用 `torch.load()` 加载检查点
- 从检查点的 `args` 字典中提取模型参数：
  - `embed_dim`：嵌入维度（默认 256）
  - `num_classes`：类别数量（必需）
  - `backbone`：骨干网络（默认 "dinov2_vitb14"）
  - `img_size`：图像尺寸（默认 128）
  - `arc_s`：ArcFace 缩放参数（默认 32.0）
  - `arc_m`：ArcFace 边界参数（默认 0.30）
- 重建 `DinoV2Embedder` 和 `ArcFaceHead` 模型
- 加载模型权重并设置为评估模式

**依赖：**
- `train_dinov2_arcface_small.DinoV2Embedder`
- `train_dinov2_arcface_small.ArcFaceHead`

### 2. 图像分类 (`classify`, `classify_batch`, `classify_image`)

**职责：** 对图像进行推理分类，支持单张和批量处理

**实现细节：**

`classify()` 函数（v0.1.4 更新）：
- 输入：嵌入器、分类头、图像张量 `[batch_size, C, H, W]`
- 使用 `@timeit(100)` 装饰器进行性能统计
- 推理流程：
  1. 通过嵌入器获取特征向量 `z`（批量）
  2. 归一化分类头权重矩阵 `W`
  3. 计算 logits：`s * linear(z, W)`（评估时不使用 margin）
  4. 计算 softmax 概率
  5. 返回批量预测类别、置信度、原始 logits（Tensor 格式）

`classify_batch()` 函数（v0.1.4 新增）：
- 输入：嵌入器、分类头、图像路径列表、变换管道、设备
- 处理流程：
  1. 批量加载图像并应用变换
  2. 堆叠为 batch tensor
  3. 调用 `classify()` 进行批量推理
  4. 处理加载失败的图像，保持结果顺序
  5. 返回结果列表，每个元素为 `(pred_class, confidence, logits)` 元组

`classify_image()` 函数（向后兼容）：
- 内部调用 `classify_batch()` 处理单张图像
- 保持原有接口不变

**性能优化：**
- 使用 `@torch.no_grad()` 禁用梯度计算
- **批量推理（v0.1.4）：** 充分利用 GPU 并行计算，显著提升处理速度
- 批量处理减少 GPU 与 CPU 之间的数据传输次数

### 3. 可视化 (`visualize_by_category`)

**职责：** 按类别分组可视化分类结果

**设计变更（v0.1.3）：**
- **之前：** 所有类别绘制在一张大图上
- **现在：** 每个类别单独绘制一张大图

**实现细节：**

1. **数据分组：**
   - 使用 `defaultdict(list)` 按预测类别分组
   - 按图像数量降序排序类别

2. **布局计算：**
   - 固定列数：4 列
   - 行数计算：`(num_images + cols - 1) // cols`（向上取整）
   - 每个类别独立计算布局

3. **文件命名：**
   - 格式：`{base_filename}_class_{class_idx}_{safe_class_name}.png`
   - 类名中的特殊字符（`/`、`\`、`:`）替换为下划线
   - 如果 `output_path` 是目录，使用默认文件名 `classification_visualization`

4. **图像显示：**
   - 使用 `matplotlib.gridspec.GridSpec` 创建网格布局
   - 第一行为类别标题
   - 后续行为图像网格（4 列）
   - 图像按置信度降序排列
   - 显示文件名（截断过长）和置信度

5. **错误处理：**
   - 图像加载失败时显示错误信息
   - 空单元格自动填充

**输出：**
- 每个类别生成一个 PNG 文件
- 打印所有生成的文件路径

## 依赖关系

### 外部依赖

```python
torch                    # PyTorch 深度学习框架
torchvision             # 图像变换工具
PIL (Pillow)            # 图像处理
matplotlib              # 可视化
tqdm                    # 进度条
```

### 内部依赖

```python
train_dinov2_arcface_small.DinoV2Embedder
train_dinov2_arcface_small.ArcFaceHead
train_dinov2_arcface_small.build_val_tfm
train_dinov2_arcface_small.CenterSquareCrop
train_dinov2_arcface_small.make_divisible
utils.utils.timeit
utils.file.walk_image
```

## 接口规范

### 函数签名

```python
def load_model(model_path: str, device: str = "cuda") -> Tuple[DinoV2Embedder, ArcFaceHead, List[str], dict]:
    """加载模型检查点"""
    pass

@torch.no_grad()
def classify(embedder, head, img_tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """批量分类推理（核心逻辑）
    
    Args:
        img_tensor: [batch_size, C, H, W] 图像张量
    
    Returns:
        pred_classes: [batch_size] 预测类别索引
        confidences: [batch_size] 置信度分数
        logits: [batch_size, num_classes] 原始 logits
    """
    pass

@torch.no_grad()
def classify_batch(embedder, head, image_paths: list, transform, device: str) -> List[Tuple[Optional[int], Optional[float], Optional[torch.Tensor]]]:
    """批量分类图像
    
    Args:
        image_paths: 图像路径列表
    
    Returns:
        结果列表，每个元素为 (pred_class, confidence, logits) 元组
    """
    pass

@torch.no_grad()
def classify_image(embedder, head, image_path: str, transform, device: str) -> Tuple[Optional[int], Optional[float], Optional[torch.Tensor]]:
    """分类单张图像（向后兼容，内部调用 classify_batch）"""
    pass

def visualize_by_category(image_results: dict, classes: list = None, output_path: str = None, max_images_per_class: int = 20) -> None:
    """可视化分类结果（每个类别单独一张图）"""
    pass
```

### 数据格式

**`image_results` 字典格式：**
```python
{
    "path/to/image1.jpg": (pred_class: int, confidence: float, logits: torch.Tensor),
    "path/to/image2.jpg": (pred_class: int, confidence: float, logits: torch.Tensor),
    ...
}
```

**检查点文件格式：**
```python
{
    "embedder": state_dict,
    "head": state_dict,
    "args": {
        "embed_dim": int,
        "num_classes": int,
        "backbone": str,
        "img_size": int,
        "arc_s": float,
        "arc_m": float
    },
    "classes": List[str]  # 可选
}
```

## 性能考虑

### 内存使用

- **模型加载：** 模型权重常驻内存
- **图像处理：** 逐张处理，避免批量加载导致内存溢出
- **可视化：** 每个类别独立生成图表，生成后立即关闭 figure 释放内存

### 计算性能

- **推理速度：** 使用 `@timeit(100)` 装饰器统计平均推理时间
- **GPU 加速：** 支持 CUDA 设备加速推理
- **批量推理（v0.1.4）：** 实现真正的批量处理，充分利用 GPU 并行计算能力，显著提升处理速度
- **批处理大小：** 通过 `--batch_size` 参数控制，默认 32，可根据 GPU 内存调整

## 已知限制

1. **内存限制：** 批量处理时，batch_size 过大可能导致 GPU 内存不足
2. **图像格式：** 依赖 PIL 支持的格式，某些特殊格式可能无法处理
3. **错误处理：** 批量处理中单个图像加载失败不影响其他图像，但会降低批次效率

## 未来改进方向

1. **动态批处理：** 根据 GPU 内存自动调整 batch_size
2. **多进程处理：** 对于 CPU 模式，可以使用多进程加速图像加载
3. **结果缓存：** 支持结果缓存，避免重复计算
4. **交互式可视化：** 支持 Web 界面或 Jupyter notebook 交互式查看
5. **性能分析：** 添加更详细的性能分析工具
6. **异步 I/O：** 使用异步 I/O 提升图像加载速度

## 测试建议

### 单元测试

- `load_model()`: 测试检查点加载和参数提取
- `classify()`: 测试推理逻辑和输出格式
- `classify_image()`: 测试图像加载和异常处理
- `visualize_by_category()`: 测试可视化生成和文件命名

### 集成测试

- 端到端测试：从图像目录到结果输出的完整流程
- 性能测试：大量图像的处理时间和内存使用
- 兼容性测试：不同检查点格式的兼容性

## 相关文件

- 模型定义：`train_dinov2_arcface_small.py`
- 数据加载：`data/unified_data_loader.py`
- 工具函数：`utils/file.py`, `utils/utils.py`

