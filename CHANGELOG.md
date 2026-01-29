# Changelog

## 0.1.19 - 2026/01/29 12:00 (smartadpole)
Auto-cleanup temp dirs and sync docs with code

- 优化：`tools/gen_sku.py` 中 `process_by_dataframe` 写出结果后在 `finally` 中自动清理临时工作目录（work_root），无需手动删除
- 变更：`tools/test_full.py` 中 `--temp_dir` 默认改为 None，未指定时使用输入 CSV 所在目录下的 `temp_cropped` 子目录，处理结束后自动删除该目录
- 修复：`test_full.py` 中临时目录名拼写 `temp_croped` → `temp_cropped`
- 优化：同步版本号与文档，并依据上述代码变更更新 docs（gen_sku/test_full 的 dev、user 及 README 中临时目录清理与 `--temp_dir` 默认行为说明）

## 0.1.18 - 2026/01/28 15:45 (smartadpole)
Fix SKU clustering API and deduplication in gen_sku

- 修复：`tools/gen_sku.py` 中 fastdup 聚类调用方式，改为使用 `fastdup.run_kmeans(input_dir=image_paths, work_dir=..., ...)`，不再通过 subset.csv + fd.run(annotations=...) 间接调用，避免聚类错误
- 修复：读取 kmeans_assignments 后按 filename 去重（保留每张图片距离最小的记录），再执行簇内采样，避免同一图片多簇导致的采样异常
- 优化：`gen_sku.py` 增加 tqdm 进度条、启动前清理 work_dirs、输出路径标准化为绝对路径，并统一使用 logger 的 `print(..., level=...)`
- 修复：`tools/test_full.py` 中 `load_image` 在图片不存在时改为输出 error 并 exit(1)，避免静默继续
- 变更：`utils/logger.py` 中 DEBUG 与 INFO 的控制台颜色对调（DEBUG 白色、INFO 蓝色）

## 0.1.17 - 2026/01/28 14:30 (smartadpole)
Add class filter mode for eval and test_full inference

- 新增：`tools/eval.py` 中 `classify`、`classify_batch` 支持可选参数 `allowed_indices`，可将预测限制在指定类别子集内（logits 掩码只保留允许类别）
- 新增：`tools/test_full.py` 在加载 `--label_file` 时根据有效类别名计算 `allowed_indices` 并传入 `classify_batch`，实现“仅预测标签文件中定义的类别”的过滤模式
- 优化：过滤模式下打印信息日志（限制到 N 个类别），无效或空 allowed_indices 时自动回退为不过滤

## 0.1.16 - 2026/01/28 12:00 (smartadpole)
Refactor gen_sku template tool CLI and add file-list mode

- 重大变更：`tools/gen_sku.py` 命令行从 `--input_dir`/`--output_dir`/`--batch`/`--single` 改为 `--input`（支持目录或文件列表）/`--output`（输出 txt 文件路径）
- 新增：`gen_sku.py` 支持文件列表模式，输入为三列格式（路径、ID、类别名），与 `eval.load_test_file` 一致，按标签组内独立聚类采样
- 新增：目录模式改为递归扫描“叶子”目录（包含图片且无子目录再含图）作为类别，支持多层嵌套
- 变更：输出由按类别复制图片到子目录改为单文件列表（每行：路径,label_id,class_name），临时工作目录位于输出文件同目录下的 `work_dirs/`
- 移除：`--batch`、`--single`、`--output_dir` 参数
- 新增：`requirements.txt` 中增加 fastdup 依赖（`fastdup>=1.0.0`）
- 优化：`README.md` 与 `docs/user/gen_sku_template_library.md` 中模板库章节与当前脚本行为一致（`--input`/`--output`、输出说明）

## 0.1.15 - 2026/01/26 18:32 (smartadpole)
Implement template library generation using fastdup clustering-based sampling

- 新增：创建 `tools/gen_sku.py` 模板库生成脚本，支持基于 fastdup 聚类的多样性采样方案
- 新增：实现 `method2_template_sampling` 函数，采用聚类中心采样法，从每个簇中提取距离中心最近的图片作为模板
- 新增：实现 `method2_hybrid_sampling` 函数，支持混合采样策略（80% 中心样本 + 20% 边缘样本），增强对极端环境的包容度
- 新增：支持批量处理多个类别文件夹，自动遍历根目录下的所有类别子目录
- 新增：支持单类别处理模式，可对单个类别文件夹生成模板库
- 新增：自动检测处理模式，根据输入目录结构自动判断是批量模式还是单类别模式
- 新增：命令行参数支持，包括 `--input_dir`、`--output_dir`、`--method`、`--num_templates`、`--edge_ratio`、`--num_em_iter` 等配置选项
- 新增：样本数量不足时的自动处理逻辑，当图片数量少于所需模板数时直接使用所有图片
- 优化：针对复杂环境（光线、烟雾、角度变化）优化聚类参数，默认迭代次数设置为 30 次
- 优化：添加完善的错误处理和日志输出，包括文件存在性检查、去重处理、复制计数等

## 0.1.14 - 2026/01/26 16:46 (smartadpole)
Implement open-set recognition evaluator with offline feature gallery

- 新增：创建 `tools/eval_open.py` 开集识别评估脚本，支持基于离线特征库的度量学习识别方案
- 新增：实现 `OpenSetEvaluator` 类，包含特征库构建、开集识别和评估功能
- 新增：特征库构建功能（`build_gallery`），从模板图提取特征并支持离群点清洗
- 新增：开集识别功能（`identify`），使用 Top-K 最近邻检索和加权投票机制
- 新增：加权投票机制（`_weighted_voting`），结合相似度、排名权重和类别大小归一化，处理类别模板不均衡问题
- 新增：离群点清洗功能（`_remove_outliers`），基于距离统计过滤异常模板特征
- 新增：双阈值开集判定逻辑，支持绝对阈值和相对增益阈值（margin threshold）判定未知类
- 新增：评估结果统计功能，包含总体准确率、分类别准确率和未知类判定数量
- 新增：命令行参数支持，包括 `--template_file`、`--threshold`、`--top_k`、`--margin_threshold`、`--outlier_threshold` 等配置选项
- 优化：复用 `tools/eval.py` 中的 `load_test_file` 和 `calculate_metrics` 函数，保持代码一致性
- 优化：复用 `train_dinov2_arcface_small.py` 中的 `DinoV2Embedder` 和 `build_val_tfm`，确保特征提取一致性

## 0.1.13 - 2026/01/23 13:52 (smartadpole)
Optimize training data augmentation for fine-grained classification tasks

- 优化：`train_dinov2_arcface_small.py` 中 `build_train_tfm` 函数收敛数据增强参数，适配细粒度分类任务（依赖细微纹理和精准色调）
- 优化：ColorJitter 参数收敛，hue 从 0.1 降至 0.01（几乎锁定色相，保护颜色关键特征），saturation 从 0.4 降至 0.2（减少饱和度差异，避免掩盖肉质特征）
- 优化：GaussianBlur sigma 上限从 4.0 降至 1.5（轻微起雾而非完全失焦，保护纹理细节）
- 优化：RandomAffine 去除错切（shear=0），防止纹理非物理拉伸，degrees 从 20 降至 15，translate 从 0.15 降至 0.1，scale 从 (0.8, 1.2) 收敛至 (0.9, 1.1)
- 优化：RandomPerspective distortion_scale 从 0.3 降至 0.1，p 从 0.5 降至 0.3（仅模拟轻微视角变化）
- 优化：RandomResizedCrop scale 从 (0.5, 1.2) 收敛至 (0.8, 1.0)，ratio 从 (0.75, 1.33) 收敛至 (0.9, 1.1)，保留更多上下文特征
- 新增：添加 RandomVerticalFlip（p=0.5），增加数据量（肉类纹理通常没有上下之分）
- 优化：RandomJPEG p 从 0.4 降至 0.2，qmin 从 30 提升至 50，qmax 从 80 提升至 95（降低压缩失真概率，保留纹理）
- 优化：RandomErasing 仅保留一个，p 从 0.4 降至 0.3，scale 从 (0.02, 0.25) 收敛至 (0.02, 0.1)，减小遮挡面积，防止遮住关键纹理
- 移除：删除第二个 RandomErasing（结构化遮挡），避免过度遮挡关键纹理区域

## 0.1.12 - 2026/01/19 19:41 (smartadpole)
Enable real-time visualization and fix bbox parsing format

- 新增：`test_full.py` 中开启 `visualize_image_with_bbox` 功能，支持实时显示图像和检测框
- 新增：加载阶段可视化功能，实时显示每一帧的大图并绘制检测框（不显示 label 和置信度）
- 新增：测评阶段可视化功能，实时显示每一帧的大图，绘制检测框、label 名字和置信度
- 修复：`parse_bbox` 函数修正 bbox 解析格式，从左上角坐标+宽高改为中心点坐标+宽高（格式：cx cy w h）
- 优化：`visualize_image_with_bbox` 函数添加图像显示功能，使用 matplotlib 实时显示图像
- 优化：图像显示时自动缩放超过 1080p（1920x1080）分辨率的图像，保持宽高比
- 优化：使用 warnings 模块和 matplotlib 配置抑制字体查找相关的警告日志
- 优化：图像显示使用非阻塞模式，每张图像显示 0.1 秒后自动关闭
- 优化：添加 PIL 版本兼容性处理，支持新旧版本的图像重采样方法

## 0.1.11 - 2026/01/19 18:41 (smartadpole)
Add category-based image saving and improve output file naming

- 新增：`test_full.py` 中新增 `generate_saved_filename` 函数，根据视频名、帧号和帧类型生成保存的文件名（格式：`video_name帧号_frame_type.jpg`）
- 新增：`test_full.py` 中新增 `--temp_save_dir` 命令行参数，支持按类别保存裁剪后的图片到指定目录
- 新增：按类别保存功能支持自动创建类别子目录，文件名包含视频名、帧号和帧类型信息，便于后续分析和查看
- 优化：`test_full.py` 中将 `--output_csv` 参数改为 `--suffix` 参数，自动根据输入 CSV 文件名生成输出文件名（格式：`{input_csv_basename}_{suffix}.csv`）
- 优化：`test_full.py` 中 `collect_all_images` 函数返回 `row_data` 供后续文件名生成使用
- 优化：`test_full.py` 中 `read_csv_auto` 函数添加文件存在性检查，提前发现文件不存在的情况
- 优化：`test_full.py` 中 `--base_dir` 参数默认值改为输入 CSV 文件所在目录，简化使用
- 新增：添加 `re` 和 `shutil` 模块导入，支持文件名处理和文件复制功能
- 新增：添加 `utils.file.mkdir_simple` 导入，自动创建输出目录

## 0.1.10 - 2026/01/16 20:36 (smartadpole)
Add security checks for label file and detailed logging for image cropping

- 新增：`test_full.py` 中添加文件安全检查工具函数（`validate_file_path`、`validate_file_size`、`validate_file_extension`、`safe_read_file`），提供路径验证、文件大小限制、扩展名检查等安全功能
- 新增：`test_full.py` 中定义安全常量（文件大小限制、允许的文件扩展名白名单）
- 新增：`test_dinov2_classification.py` 中 `load_label_file` 函数添加完整的安全检查，包括路径规范化（防止目录遍历攻击）、文件存在性检查、文件类型验证（仅允许 .txt, .csv）、文件大小限制（10MB）、文件可读性检查
- 新增：`load_label_file` 函数添加行数限制（最多 100000 行，防止 DoS 攻击）
- 新增：`load_label_file` 函数添加 Class ID 验证（必须 >= 0，最大 100000，防止内存耗尽）
- 优化：`test_full.py` 中 `collect_all_images` 函数添加详细的日志记录，包括每个处理步骤的 DEBUG 级别日志
- 优化：`collect_all_images` 函数添加失败原因统计和分类（image_not_found、bbox_parse_error、save_error、missing_column、empty_image_name 等）
- 优化：`collect_all_images` 函数输出失败统计摘要和前 10 个失败样本的详细信息，便于快速定位问题
- 优化：`test_full.py` 中 `parse_bbox`、`load_image`、`crop_image_by_bbox`、`save_cropped_image` 函数添加详细的 DEBUG 级别日志记录
- 优化：`load_label_file` 函数改进错误处理，添加更详细的警告信息，包括文件路径、行号、具体错误原因等

## 0.1.9 - 2026/01/16 19:29 (smartadpole)
Add visualization feature for test_full.py and create requirements.txt

- 新增：`test_full.py` 中添加 `visualize_image_with_bbox` 函数，支持在原始图片上绘制 bbox、标签和置信度信息
- 新增：`test_full.py` 中添加 `--visualize` 命令行参数，控制是否进行可视化
- 新增：`test_full.py` 中添加 `--vis_output_dir` 命令行参数，指定可视化图片输出目录
- 优化：`test_full.py` 中 `collect_all_images` 函数返回原始图片路径和 bbox 信息，供可视化使用
- 新增：可视化功能支持跨平台字体加载（Windows/Linux/macOS）
- 新增：可视化功能兼容不同版本的 PIL，支持 textbbox 和 textsize 方法
- 新增：创建 `requirements.txt` 文件，包含所有项目依赖及版本要求
- 优化：`README.md` 更新依赖列表，添加快速安装说明

## 0.1.8 - 2026/01/16 17:41 (smartadpole)
Add batch inference tool for CSV-based image classification with bbox cropping

- 新增：创建 `tools/test_full.py` 批量推理工具，支持从 CSV 文件读取图片路径和 bbox 信息进行批量分类
- 新增：`test_full.py` 中实现 `parse_bbox` 函数，解析归一化 bbox 坐标（格式：x y w h）并转换为像素坐标
- 新增：`test_full.py` 中实现 `crop_image_by_bbox` 函数，根据 bbox 坐标从原始图片中裁剪出目标区域
- 新增：`test_full.py` 中实现 `collect_all_images` 函数，批量收集和裁剪所有需要处理的图片
- 新增：`test_full.py` 支持处理 4 种类型的图片：取走目标的第一帧、取走目标的过线帧、放回目标的过线帧、放回目标的静止帧
- 新增：`test_full.py` 输出 CSV 文件包含 8 列预测结果（每张图片的类别名称和置信度）
- 优化：`test_full.py` 复用 `eval.py` 中的 `load_model`、`classify_batch`、`build_val_tfm` 函数，避免重复实现
- 优化：`test_full.py` 复用 `test_dinov2_classification.py` 中的 `load_label_file` 函数加载类别名称映射
- 优化：`test_full.py` 使用批量处理策略，先收集所有图片再批量推理，提高处理效率
- 优化：`test_full.py` 输出类别名称而不是类别ID，提升结果可读性
- 新增：`test_full.py` 支持通过 `--label_file` 参数指定类别名称映射文件
- 新增：`test_full.py` 支持通过 `--base_dir` 参数指定图片根目录，支持相对路径解析
- 新增：`test_full.py` 添加完善的错误处理，处理图片缺失、bbox 解析失败等情况

## 0.1.7 - 2026/01/16 15:23 (smartadpole)
Add evaluation script with accuracy metrics and per-class visualization

- 新增：创建 `tools/eval.py` 精度测评脚本，支持 DINOv2 ArcFace 模型的精度评估
- 新增：`eval.py` 中实现 `load_test_file` 函数，支持从 txt 文件读取测试数据（三列格式：图片路径、标签ID、类别名称）
- 新增：`eval.py` 中实现 `calculate_metrics` 函数，计算总体精度和分类别精度，并统计正确数、错误数、总数
- 新增：`eval.py` 中实现 `visualize_results` 函数，按类别分组可视化，每个类别生成独立图片
- 新增：可视化功能显示每个类别的前 10 个正确样本（按置信度排序）和所有错误样本
- 新增：统计结果输出包含总体和分类别的精度、正确数、错误数、总数等详细信息
- 新增：`eval.py` 中实现批量推理功能，支持通过 `--batch_size` 参数配置批量大小
- 新增：结果保存为文本文件，包含完整的精度统计信息
- 优化：`eval.py` 中测试文件解析逻辑支持制表符和空格分隔符，正确处理包含空格的图片路径

## 0.1.6 - 2026/01/16 14:42 (smartadpole)
Add label file support for classification test tool

- 新增：`test_dinov2_classification.py` 中新增 `load_label_file` 函数，支持从文本文件读取类别标签
- 新增：`test_dinov2_classification.py` 中添加 `--label` 命令行参数，支持指定标签文件路径
- 优化：`load_label_file` 函数支持多种分隔符格式（制表符、逗号、空格），提高文件格式兼容性
- 优化：`test_dinov2_classification.py` 中主程序支持从外部标签文件覆盖 checkpoint 中的类别名称，提升灵活性
- 优化：`load_label_file` 函数添加完善的错误处理和警告信息，提高健壮性

## 0.1.5 - 2026/01/16 01:06 (smartadpole)
Enhance training data augmentation with advanced transformations

- 优化：`train_dinov2_arcface_small.py` 中 `build_train_tfm` 函数升级数据增强策略，添加仿射变换和透视变换，模拟视角变换与随机摆放
- 优化：增强烟雾与环境干扰模拟，包括随机 JPEG 压缩、色彩抖动、大核高斯模糊和随机灰度化，提升模型对复杂光照条件的鲁棒性
- 优化：改进遮挡处理，使用双重随机擦除策略，分别模拟手部或较大物体遮挡（随机噪点块）和结构化遮挡（黑色或深色实心块）
- 优化：调整数据增强参数，包括随机裁剪缩放范围、仿射变换参数和透视变换扭曲程度，提升数据增强效果

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

