# CSV 批量推理工具技术文档

## 概述

本文档描述 `tools/test_full.py` 的技术实现细节，面向开发者和维护者。该工具用于从 CSV 文件中读取图片路径和 bbox 信息，批量裁剪图片并进行分类推理，最后将结果输出到新的 CSV 文件。

## 架构设计

### 模块结构

```
test_full.py
├── 数据解析模块
│   ├── parse_bbox()          # 解析归一化 bbox 坐标
│   └── crop_image_by_bbox()  # 根据 bbox 裁剪图片
├── 图片收集模块
│   ├── collect_all_images()  # 批量收集和裁剪所有图片
│   └── save_cropped_image()  # 保存裁剪后的临时图片
├── 文件名生成模块
│   └── generate_saved_filename()  # 根据视频名、帧号和帧类型生成保存的文件名
├── 可视化模块
│   └── visualize_image_with_bbox()  # 在原始图片上绘制 bbox、标签和置信度
├── 推理模块（复用）
│   ├── load_model()          # 从 eval.py 导入
│   ├── classify_batch()      # 从 eval.py 导入（支持 allowed_indices 类别过滤）
│   └── build_val_tfm()       # 从 eval.py 导入
└── 主程序
    └── main()
```

### 数据流

```
CSV 文件 → 读取所有行
    ↓
遍历每行的 4 个图片字段 → crop_image_by_bbox() → 裁剪图片
    ↓
保存到临时目录 → collect_all_images() → 收集所有任务
    ↓
批量推理 (batch_size) → classify_batch() → 批量分类结果
    ↓
映射结果回 DataFrame → 添加 8 列结果 → 输出 CSV 文件
```

## 核心功能实现

### 1. Bbox 解析 (`parse_bbox`)

**职责：** 将归一化的 bbox 字符串转换为像素坐标

**输入格式：**
- `bbox_str`: 归一化坐标字符串，格式为 "cx cy w h"（空格分隔，中心点坐标+宽高）
- 坐标值范围：[0, 1]，表示相对于图片尺寸的比例
- `cx, cy`: 中心点坐标（归一化）
- `w, h`: 宽度和高度（归一化）

**实现细节：**
- 支持空格分隔的 4 个浮点数
- 转换为像素坐标：
  - `cx_pixel = cx_norm * img_width`
  - `cy_pixel = cy_norm * img_height`
  - `w_pixel = w_norm * img_width`
  - `h_pixel = h_norm * img_height`
- 将中心点转换为左上角坐标：
  - `x = cx - w / 2`
  - `y = cy - h / 2`
- 边界检查：确保 bbox 在图片范围内
- 最小尺寸保证：宽度和高度至少为 1 像素

**错误处理：**
- 空值或无效格式返回 `None`
- 解析失败时打印警告信息

### 2. 图片裁剪 (`crop_image_by_bbox`)

**职责：** 从原始图片中根据 bbox 裁剪出目标区域

**实现细节：**
- 支持相对路径和绝对路径
- 如果路径为相对路径且提供了 `base_dir`，则拼接完整路径
- 使用 PIL 打开图片并转换为 RGB 模式
- 调用 `parse_bbox` 获取像素坐标
- 使用 PIL 的 `crop()` 方法裁剪：`(left, top, right, bottom)`

**错误处理：**
- 图片不存在返回 `None`
- bbox 解析失败返回 `None`
- 图片打开失败时打印警告并返回 `None`

### 3. 批量图片收集 (`collect_all_images`)

**职责：** 从 DataFrame 中收集所有需要处理的图片并批量裁剪

**处理的图片字段：**
1. `take_first_image_name` + `take_first_bbox` → `take_first`
2. `take_cross_image_name` + `take_cross_bbox` → `take_cross`
3. `return_image_name` + `return_bbox` → `return`
4. `return_static_image_name` + `return_static_bbox` → `return_static`

**实现细节：**
- 遍历 DataFrame 的每一行
- 对每行的 4 个图片字段进行处理
- 临时文件命名格式：`row_{row_idx}_{prefix}_{original_filename}`
- 返回成功任务列表和失败任务列表，包含原始图片路径和 bbox 信息供可视化使用

**输出：**
- `tasks`: `List[Tuple[row_idx, field_prefix, temp_path, orig_img_path, bbox_str, row_data]]`
- `failed_tasks`: `List[Tuple[row_idx, field_prefix, reason]]`

### 4. 可视化功能 (`visualize_image_with_bbox`)

**职责：** 在原始图片上绘制 bbox、标签和置信度信息，并实时显示图像

**实现细节：**
- 加载原始图片并解析 bbox 坐标（支持中心点+宽高格式）
- 使用 PIL ImageDraw 在图片上绘制红色边界框
- 可选显示标签文本和置信度（格式：`label: confidence`）
- 自动调整字体大小和文本位置，确保文本在图片范围内
- 支持跨平台字体加载（Windows/Linux/macOS）
- 兼容不同版本的 PIL（支持 textbbox 和 textsize 方法）
- 图像显示前自动缩放超过 1080p（1920x1080）分辨率的图像，保持宽高比
- 使用 matplotlib 实时显示图像（非阻塞模式）

**字体加载策略：**
- Windows: 尝试加载 Arial、Microsoft YaHei 等系统字体
- Linux: 尝试加载 DejaVu Sans、Liberation Sans 等字体
- macOS: 尝试加载 Helvetica、Arial 等字体
- 如果系统字体不可用，回退到默认字体

**图像显示优化：**
- 超过 1080p 的图像自动缩放到 1080p 以内，保持宽高比
- 使用 LANCZOS 重采样保证缩放质量
- 兼容新旧版本的 PIL（自动回退到 `Image.LANCZOS`）
- 使用 matplotlib 非阻塞模式显示，每张图像显示 0.1 秒后自动关闭
- 抑制 matplotlib 字体查找相关的警告日志

**可视化效果：**
- 红色边界框（bbox）
- 黑色背景的白色文本标签（可选）
- 文本显示格式：`类别名称: 置信度`（置信度保留 3 位小数）

**使用场景：**
- **加载阶段**：只显示图像和检测框（`show_label=False`），用于检查 bbox 是否正确
- **测评阶段**：显示图像、检测框、标签和置信度（`show_label=True`），用于查看分类结果

### 5. 类别名称映射

**职责：** 将预测的类别ID转换为类别名称

**实现细节：**
- 复用 `test_dinov2_classification.load_label_file` 函数
- 支持从标签文件加载类别名称映射
- 如果类别ID不在映射中，使用 `Class_{id}` 作为回退

**标签文件格式：**
```
label_id class_name
0 苹果
1 香蕉
2 橙子
```

支持制表符或空格分隔。

### 6. 文件名生成 (`generate_saved_filename`)

**职责：** 根据视频名、帧号和帧类型生成保存的文件名

**实现细节：**
- 支持从多种可能的列名中自动识别视频名和帧号（优先级：参数指定 > 常见列名）
- 视频名列名优先级：`video_name` > `video_path` > `video` > `vid_name` > `vid_path`
- 帧号列名优先级：`event_frame` > `cross_frame` > `frame_number` > `frame_num` > `cross_line_frame` > `frame`
- 自动提取视频文件名（如果视频名是路径，提取文件名部分）
- 帧类型映射：
  - `take_first` → `取走目标的第一帧`
  - `take_cross` → `取走目标的过线帧`
  - `return` → `放回目标的过线帧`
  - `return_static` → `放回目标的静止帧`
- 文件名格式：`{video_name}_{frame_number}_{frame_type}.jpg`
- 自动清理文件名中的非法字符（`<>:"/\|?*`）

**错误处理：**
- 如果找不到视频名，使用 `unknown_video` 作为默认值
- 如果找不到帧号，使用 `none` 作为默认值

### 7. 按类别保存图片

**职责：** 将裁剪后的图片按预测类别保存到指定目录

**实现细节：**
- 如果指定 `--temp_save_dir` 参数，会在该目录下创建以模型版本命名的子目录
- 在每个版本目录下，按类别名称创建子目录
- 使用 `generate_saved_filename` 生成文件名，包含视频名、帧号和帧类型信息
- 从临时目录复制裁剪后的图片到对应的类别目录

**目录结构：**
```
{temp_save_dir}/
  └── {model_version}/
      ├── {class_name_1}/
      │   ├── video1_123_取走目标的第一帧.jpg
      │   └── ...
      ├── {class_name_2}/
      │   └── ...
      └── ...
```

## 依赖关系

### 外部依赖

```python
pandas          # CSV 文件读写
torch           # PyTorch 深度学习框架
PIL (Pillow)    # 图像处理和可视化绘制
tqdm            # 进度条显示
```

### 内部依赖

```python
eval.load_model                    # 模型加载
eval.classify_batch                # 批量推理
eval.build_val_tfm                 # 图像变换管道
test_dinov2_classification.load_label_file  # 类别名称加载
utils.logger.logger_manager       # 日志管理
utils.file.mkdir_simple           # 目录创建工具
```

## 接口规范

### 函数签名

```python
def parse_bbox(bbox_str: str, img_width: int, img_height: int) -> Optional[Tuple[int, int, int, int]]:
    """解析归一化 bbox 字符串为像素坐标
    
    Args:
        bbox_str: 归一化坐标字符串 "cx cy w h"（中心点坐标+宽高）
        img_width: 图片宽度（像素）
        img_height: 图片高度（像素）
    
    Returns:
        (x, y, w, h) 像素坐标元组（左上角坐标+宽高），或 None 如果解析失败
    """
    pass

def crop_image_by_bbox(image_path: str, bbox_str: str, base_dir: str = None) -> Optional[Image.Image]:
    """根据 bbox 裁剪图片
    
    Args:
        image_path: 图片文件路径（相对或绝对）
        bbox_str: 归一化 bbox 字符串
        base_dir: 基础目录，用于解析相对路径
    
    Returns:
        PIL Image 对象，或 None 如果失败
    """
    pass

def collect_all_images(df: pd.DataFrame, base_dir: str, temp_dir: str, visualize: bool = False) -> Tuple[List, List]:
    """收集所有需要处理的图片并批量裁剪
    
    Args:
        df: 输入 DataFrame
        base_dir: 图片基础目录
        temp_dir: 临时文件保存目录
        visualize: 是否进行可视化（未使用，保留用于兼容性）
    
    Returns:
        (tasks, failed_tasks) 元组
        tasks: List[Tuple[row_idx, field_prefix, temp_path, orig_img_path, bbox_str, row_data]]
        failed_tasks: List[Tuple[row_idx, field_prefix, reason]]
    """
    pass

def generate_saved_filename(row: pd.Series, prefix: str, video_name_col: str = None, 
                            cross_frame_col: str = None) -> str:
    """根据视频名、帧号和帧类型生成保存的文件名
    
    Args:
        row: pandas Series 包含行数据
        prefix: 字段前缀（take_first, take_cross, return, return_static）
        video_name_col: 视频名列名（可选，默认尝试常见列名）
        cross_frame_col: 帧号列名（可选，默认尝试常见列名）
    
    Returns:
        生成的文件名字符串（格式：video_name帧号_frame_type.jpg）
    """
    pass

def visualize_image_with_bbox(image_path: str, bbox_str: str, label: str = None, confidence: float = None,
                               base_dir: str = None, show_label: bool = True, display: bool = True) -> Optional[Image.Image]:
    """在原始图片上绘制 bbox、标签和置信度，并实时显示图像
    
    Args:
        image_path: 原始图片路径（相对或绝对）
        bbox_str: 归一化 bbox 字符串 "cx cy w h"（中心点坐标+宽高）
        label: 预测的类别名称（可选，仅在 show_label=True 时使用）
        confidence: 预测置信度（可选，仅在 show_label=True 时使用）
        base_dir: 基础目录，用于解析相对路径
        show_label: 是否显示标签和置信度文本（默认：True）
        display: 是否使用 matplotlib 实时显示图像（默认：True）
    
    Returns:
        PIL Image 对象（已绘制 bbox 和文本），或 None 如果失败
    """
    pass
```

### 命令行参数

```bash
--input_csv      # 输入 CSV 文件路径（必需）
--suffix         # 输出 CSV 文件后缀名（可选，默认：label），输出文件名为 {input_csv_basename}_{suffix}.csv
--model_path     # 模型检查点文件路径（必需）
--label_file     # 类别名称映射文件路径（可选）
--base_dir       # 图片基础目录（可选，默认：输入 CSV 文件所在目录）
--device         # 推理设备：cuda 或 cpu（默认：cuda）
--batch_size     # 批量推理大小（默认：32）
--temp_dir       # 临时文件目录（默认：temp_cropped）
--temp_save_dir  # 按类别保存裁剪图片的目录（可选）
--visualize      # 是否进行可视化（默认：False）。启用后会在加载阶段和测评阶段实时显示图像
```

### 输入 CSV 格式

**必需列：**
- `take_first_image_name`: 取走目标的第一帧图片路径
- `take_first_bbox`: 第一帧的 bbox（格式：cx cy w h，中心点坐标+宽高）
- `take_cross_image_name`: 取走目标的过线帧图片路径
- `take_cross_bbox`: 过线帧的 bbox（格式：cx cy w h）
- `return_image_name`: 放回目标的过线帧图片路径
- `return_bbox`: 放回过线帧的 bbox（格式：cx cy w h）
- `return_static_image_name`: 放回目标的静止帧图片路径
- `return_static_bbox`: 静止帧的 bbox（格式：cx cy w h）

**其他列：** 保留在输出 CSV 中

### 输出 CSV 格式

**新增列（8 列）：**
- `take_first_image_label`: 第一帧预测类别名称
- `take_first_image_confidence`: 第一帧预测置信度
- `take_cross_image_label`: 过线帧预测类别名称
- `take_cross_image_confidence`: 过线帧预测置信度
- `return_image_label`: 放回过线帧预测类别名称
- `return_image_confidence`: 放回过线帧预测置信度
- `return_static_image_label`: 静止帧预测类别名称
- `return_static_image_confidence`: 静止帧预测置信度

## 业务逻辑流程

### 主流程

1. **初始化阶段**
   - 解析命令行参数
   - 设置设备（CUDA/CPU）
   - 加载类别名称映射（如果提供）；若存在有效类别名则计算 `allowed_indices`，推理时仅预测这些类别（过滤模式）
   - 加载模型和构建变换管道

2. **数据读取阶段**
   - 读取输入 CSV 文件
   - 验证必需列是否存在

3. **图片处理阶段**
   - 遍历所有行，收集需要处理的图片
   - 批量裁剪图片并保存到临时目录
   - 统计成功和失败的任务

4. **批量推理阶段**
   - 按 `batch_size` 分批处理所有裁剪后的图片
   - 调用 `classify_batch(..., allowed_indices=allowed_indices)` 进行批量推理（过滤模式下仅预测 label_file 中定义的类别）
   - 显示进度条

5. **结果映射阶段**
   - 将推理结果映射回原始 DataFrame
   - 使用类别名称映射将类别ID转换为名称
   - 处理失败的任务（设置为 None）
   - 如果指定 `--temp_save_dir`，按类别保存裁剪后的图片

6. **按类别保存阶段**（可选）
   - 如果启用 `--temp_save_dir` 参数，为每张成功分类的图片按类别保存
   - 使用 `generate_saved_filename` 生成包含视频名、帧号和帧类型的文件名
   - 从临时目录复制图片到对应的类别目录

7. **可视化阶段**（可选）
   - 如果启用 `--visualize` 参数，在加载阶段和测评阶段实时显示可视化结果
   - **加载阶段**：实时显示每一帧的大图，绘制检测框（不显示 label 和置信度）
   - **测评阶段**：实时显示每一帧的大图，绘制检测框、label 名字和置信度
   - 超过 1080p 的图像自动缩放到 1080p 以内显示
   - 使用 matplotlib 非阻塞模式，每张图像显示 0.1 秒后自动关闭

8. **输出阶段**
   - 根据 `--suffix` 参数自动生成输出 CSV 文件名（格式：`{input_csv_basename}_{suffix}.csv`）
   - 将结果保存到输出 CSV 文件
   - 使用 UTF-8-BOM 编码确保 Excel 正确显示中文

## 设计决策

### 为什么复用 eval.py 的函数？

**决策：** 直接导入 `eval.py` 中的 `load_model`、`classify_batch`、`build_val_tfm` 函数，而不是重新实现。

**理由：**
1. **DRY 原则：** 避免重复代码，减少维护成本
2. **一致性：** 确保推理逻辑与评估脚本完全一致
3. **可维护性：** 如果推理逻辑需要更新，只需修改一处
4. **类别过滤：** 使用 `--label_file` 时，将有效类别索引作为 `allowed_indices` 传入 `classify_batch`，预测仅在该子集内取 argmax，避免模型输出头大于标签集时的越界或无关类别被选中。

### 为什么先收集所有图片再批量推理？

**决策：** 采用两阶段处理：先收集和裁剪所有图片，再批量推理。

**理由：**
1. **效率：** 批量推理可以充分利用 GPU 并行计算能力
2. **内存管理：** 可以更好地控制内存使用，避免同时处理过多图片
3. **错误处理：** 可以统一处理失败的图片，不影响其他图片的处理

### 为什么使用临时文件？

**决策：** 将裁剪后的图片保存到临时目录，而不是直接在内存中处理。

**理由：**
1. **兼容性：** `classify_batch` 函数接受文件路径列表，而不是 PIL Image 对象
2. **调试：** 可以检查裁剪后的图片是否正确
3. **灵活性：** 如果处理中断，可以从中断点继续

### 为什么按类别保存图片？

**决策：** 提供 `--temp_save_dir` 参数，支持按预测类别保存裁剪后的图片。

**理由：**
1. **数据分析：** 便于按类别查看和分析裁剪后的图片
2. **质量检查：** 可以快速检查每个类别的预测结果是否正确
3. **样本收集：** 可以收集每个类别的样本用于后续训练或分析
4. **可追溯性：** 文件名包含视频名、帧号和帧类型信息，便于追溯原始数据

### 为什么自动生成输出文件名？

**决策：** 将 `--output_csv` 参数改为 `--suffix` 参数，自动根据输入文件名生成输出文件名。

**理由：**
1. **简化使用：** 用户只需指定后缀，无需手动构造输出文件名
2. **避免错误：** 减少因手动输入路径错误导致的问题
3. **一致性：** 输出文件名与输入文件名保持关联，便于识别

## 性能考虑

### 内存使用

- **模型加载：** 模型权重常驻内存
- **图片处理：** 逐张裁剪，避免同时加载过多图片
- **临时文件：** 裁剪后的图片保存在磁盘，处理完成后可以删除

### 计算性能

- **批量推理：** 使用 `batch_size` 参数控制批量大小，充分利用 GPU
- **进度显示：** 使用 tqdm 显示处理进度
- **并行处理：** 批量推理时 GPU 可以并行处理多张图片

### 优化建议

1. **临时文件清理：** 处理完成后可以自动删除临时文件（当前代码中已注释）
2. **缓存机制：** 如果同一张图片被多次使用，可以缓存裁剪结果
3. **多进程裁剪：** 对于 CPU 模式，可以使用多进程加速图片裁剪

## 已知限制

1. **临时文件占用：** 处理大量图片时，临时目录可能占用大量磁盘空间
2. **路径解析：** 相对路径解析依赖于 `base_dir` 参数，如果路径格式不一致可能导致失败
3. **bbox 格式：** 仅支持归一化坐标格式（中心点+宽高：cx cy w h），不支持像素坐标格式
4. **错误恢复：** 如果处理中断，需要重新运行整个流程

## 未来改进方向

1. **增量处理：** 支持从上次中断点继续处理
2. **结果缓存：** 支持结果缓存，避免重复计算
3. **多格式支持：** 支持更多 bbox 格式（像素坐标、COCO 格式等）
4. **并行裁剪：** 使用多进程加速图片裁剪
5. **内存模式：** 支持直接在内存中处理，避免临时文件
6. **可视化增强：** 支持自定义颜色、字体大小、文本位置等可视化选项

## 测试建议

### 单元测试

- `parse_bbox()`: 测试各种 bbox 格式的解析
- `crop_image_by_bbox()`: 测试图片裁剪和路径解析
- `collect_all_images()`: 测试批量图片收集

### 集成测试

- 端到端测试：从 CSV 文件到输出 CSV 文件的完整流程
- 错误处理测试：测试各种错误情况（图片缺失、bbox 无效等）
- 性能测试：大量图片的处理时间和内存使用

## 相关文件

- 推理功能：`tools/eval.py`
- 类别名称加载：`tools/test_dinov2_classification.py`
- 模型定义：`train_dinov2_arcface_small.py`
- 日志工具：`utils/logger.py`

