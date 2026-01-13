# Changelog

## 0.1.0 - 2026/01/14 03:43 (smartadpole)
Improve data loader flexibility and command-line interface

- 优化：`data_loader.py` 中 `load_data` 函数支持自动查找 `train.txt` 和 `val.txt` 文件，无需显式指定路径
- 优化：`MyDataset` 类支持逗号和空格两种分隔符格式的标签文件
- 优化：改进图像路径拼接逻辑，使用 `os.path.join` 确保跨平台兼容性
- 变更：`main.py` 中 `--train_path` 和 `--test_path` 参数改为可选，支持自动发现数据文件
- 变更：`main.py` 中 `--dataset` 默认值改为 `other`，`--image_path` 和 `--weight_path` 改为必需参数
- 新增：`data_loader.py` 中添加 `use_absolute_path` 参数支持灵活的路径处理方式

