# Changelog

## 0.1.4 - 2026/01/15 17:36 (smartadpole)
Add batch inference support and CSV output format for classification test tool

- 优化：`test_dinov2_classification.py` 中 `classify` 函数支持批量推理，提升 GPU 推理性能
- 新增：`test_dinov2_classification.py` 中新增 `classify_batch` 函数，实现批量图像加载和推理
- 优化：`test_dinov2_classification.py` 中主程序改为批量处理图像，充分利用 GPU 并行计算能力
- 变更：`test_dinov2_classification.py` 中结果文件格式从文本文件改为标准 CSV 格式，便于数据分析
- 新增：`test_dinov2_classification.py` 中添加 CSV 模块导入，使用 `csv.writer` 确保格式正确性

## 0.1.3 - 2026/01/15 17:10 (smartadpole)
Add DINOv2 classification test tool and improve visualization

- 新增：创建 `tools/test_dinov2_classification.py` 测试脚本，支持 DINOv2 ArcFace 模型的图像分类和结果可视化
- 新增：`test_dinov2_classification.py` 中实现图像分类功能，支持批量处理嵌套目录中的图像
- 新增：`test_dinov2_classification.py` 中实现分类结果文本导出功能，生成 CSV 格式的结果文件
- 优化：`visualize_by_category` 函数改为每个类别单独绘制一张大图，提高可视化清晰度
- 优化：`test_dinov2_classification.py` 从 `train_dinov2_arcface_small.py` 导入 `CenterSquareCrop` 和 `make_divisible`，移除重复定义
- 修复：修正 `utils/__init__.py`、`utils/file.py`、`utils/logger.py` 中的导入路径，从 `review_core.utils` 改为 `utils`
- 修复：修正 `utils/config.py` 中 `dotenv` 导入，添加异常处理避免缺少依赖时崩溃
- 修复：修正 `test_dinov2_classification.py` 中结果文件写入格式，移除多余的括号

## 0.1.2 - 2026/01/15 01:33 (smartadpole)
Enhance training monitoring and model saving in train_dinov2_arcface_small

- 重大变更：`train_dinov2_arcface_small.py` 中 `--num_classes` 参数改为必需参数，移除自动推断类别数的逻辑
- 新增：`train_dinov2_arcface_small.py` 中添加训练准确率计算功能，实时监控训练过程
- 新增：添加结果日志文件 `results_test.txt`，记录每个 epoch 的训练准确率、验证准确率和训练损失
- 优化：分别保存最佳训练准确率模型和最佳验证准确率模型，保存路径分别为 `best_train_dinov2_arcface_small.pt` 和 `best_val_dinov2_arcface_small.pt`
- 优化：改进训练日志输出格式，统一显示训练准确率、验证准确率和训练损失

## 0.1.1 - 2026/01/14 16:47 (smartadpole)
Unify dataset loading across multiple formats

- 新增：创建 `unified_data_loader.py` 统一数据加载模块，支持 ImageFolder 和文本列表两种数据集格式
- 新增：`unified_data_loader.py` 中实现自动格式检测功能，根据目录结构或文件自动识别数据集格式
- 新增：`TextListDataset` 类支持从文本文件加载数据集，兼容逗号和空格分隔符
- 优化：`train_dinov2_arcface_small.py` 使用统一数据加载器，支持多种数据集格式
- 优化：`data_loader.py` 中的 `load_data` 函数内部使用统一数据加载器，保持向后兼容
- 优化：统一数据加载器支持显式指定路径或自动检测，提高灵活性
- 优化：移除冗余的 `image_root` 参数，统一使用 `data_root` 作为图片根目录，简化接口
- 变更：`train_dinov2_arcface_small.py` 新增 `--train_path`、`--val_path` 参数支持灵活的数据集路径配置

## 0.1.0 - 2026/01/14 03:43 (smartadpole)
Improve data loader flexibility and command-line interface

- 优化：`data_loader.py` 中 `load_data` 函数支持自动查找 `train.txt` 和 `val.txt` 文件，无需显式指定路径
- 优化：`MyDataset` 类支持逗号和空格两种分隔符格式的标签文件
- 优化：改进图像路径拼接逻辑，使用 `os.path.join` 确保跨平台兼容性
- 变更：`main.py` 中 `--train_path` 和 `--test_path` 参数改为可选，支持自动发现数据文件
- 变更：`main.py` 中 `--dataset` 默认值改为 `other`，`--image_path` 和 `--weight_path` 改为必需参数
- 新增：`data_loader.py` 中添加 `use_absolute_path` 参数支持灵活的路径处理方式

