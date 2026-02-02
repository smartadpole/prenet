# 数据集制作

本文档介绍如何使用 `scripts/gen_all_dataset.sh` 从已标注目录生成训练/验证清单、标签映射与统计文件。

## 适用场景
- 已有图片数据集，且每个类别存放在独立目录中,并且用 `auto classify` 标注工具处理过，
- 需要生成 `train.txt` / `val.txt` 等清单文件，供训练与评估使用,
- 工具仅会处理所有状态为 “已处理” 的类别，详情见标注工具文档说明 [auto_classify](http://git-app.haidilao.com/sunh8/autolabel)  

可通过是否有 `.processed_folders.json` 识别是否经过标注工具处理，

## 依赖与环境
- Linux/macOS Bash 环境（脚本为 bash）
- `jq`（读取 `.processed_folders.json`）
- `awk` / `sed` / `sort` / `uniq` / `shuf`

## 输入目录要求
脚本会递归查找 `SEARCH_ROOT` 下的 `.processed_folders.json`，并从其中读取 `folders` 列表，作为“已标注目录”的相对路径。随后在每个目录中仅扫描一层（`-maxdepth 1`）的图片文件：
- 支持后缀：`.jpg` / `.jpeg` / `.png`
- 目录名会被解析为类别名

目录名解析规则（与脚本一致）：
1. 目录名包含下划线 `_`：取最后一段作为类别名  
2. 目录名为纯数字：按文件夹路径生成 `待定N`  
3. 其它情况：目录名即类别名

## 运行方式
```bash
bash scripts/gen_all_dataset.sh <search_root>
```
输入是图片根目录，若不传参，默认 `search_root=.` 即当前目录。

## 输出文件说明
脚本会在当前工作目录生成以下文件：

- `output.txt`  
  每行 3 列：`绝对路径,标签ID,类别名`
- `output_train.txt` / `output_test.txt`  
  由 `output.txt` 按 7:3 随机拆分
- `train.txt` / `val.txt`  
  软链接，分别指向 `output_train.txt` / `output_test.txt`
- `count_file.txt` / `count_train.txt` / `count_val.txt`  
  基于完整路径统计各目录样本量（便于核对目录级样本量）
- `count.txt`  
  按“类别名”统计样本量
- `label.txt`  
  标签映射（`label_id,类别名`），已去重并排序

## 输出示例（节选）
以下为执行 `head * -n 2` 的示例片段，展示各文件的前两行：

```text
==> count_file.txt <==
   2581 /media/hao/image/Haidilao/dapaidang/image/classify/images/2025-12-24-all-stall-result3/自选-锅底鲜汤-新/五指毛桃锅,105,五指毛桃锅
    688 /media/hao/image/Haidilao/dapaidang/image/classify/images/2025-12-24-all-stall-result3/自选-鲜切工坊-新/鲜切牛雪花,119,鲜切牛雪花

==> count_train.txt <==
   1831 /media/hao/image/Haidilao/dapaidang/image/classify/images/2025-12-24-all-stall-result3/自选-锅底鲜汤-新/五指毛桃锅,105,五指毛桃锅
    477 /media/hao/image/Haidilao/dapaidang/image/classify/images/2025-12-24-all-stall-result3/自选-鲜切工坊-新/鲜切牛雪花,119,鲜切牛雪花

==> count.txt <==
   2581 五指毛桃锅,105,五指毛桃锅
   1718 生蚝,53,生蚝

==> count_val.txt <==
    750 /media/hao/image/Haidilao/dapaidang/image/classify/images/2025-12-24-all-stall-result3/自选-锅底鲜汤-新/五指毛桃锅,105,五指毛桃锅
    211 /media/hao/image/Haidilao/dapaidang/image/classify/images/2025-12-24-all-stall-result3/自选-鲜切工坊-新/鲜切牛雪花,119,鲜切牛雪花

==> label.txt <==
1,中山脆皖
2,安格斯精品肥牛

==> output_test.txt <==
/media/hao/image/Haidilao/dapaidang/image/classify/images/2025-12-24-all-stall-result3/自选-锅底鲜汤-新/花椒鸡锅/01010000411000000.mp4_10456_取走目标的过线帧.jpg,111,花椒鸡锅
/media/hao/image/Haidilao/dapaidang/image/classify/images/2025-12-24-all-stall-result3/自选-田园时蔬-新/脆脆毛肚/01010000413000000.mp4_60963_放回目标的静止帧.jpg,26,脆脆毛肚

==> output_train.txt <==
/media/hao/image/Haidilao/dapaidang/image/classify/images/2025-12-24/sku_train_李玉乐/sku_自选-锅底鲜汤-新/红酸汤/01000004946000000_41382_51.jpg,144,红酸汤锅
/media/hao/image/Haidilao/dapaidang/image/classify/images/2025-12-24/sku_train_孙昊/自选-甄选海鲜2-新/淡菜/01010000462000000_59280_55.jpg,63,淡菜

==> output.txt <==
/media/hao/image/Haidilao/dapaidang/image/classify/images/2025-12-24-all-stall-result3/自选-刺身小吃鲜切工坊-新/中山脆皖/01010000415000000.mp4_33603_取走目标的第一帧.jpg,1,中山脆皖
/media/hao/image/Haidilao/dapaidang/image/classify/images/2025-12-24-all-stall-result3/自选-刺身小吃鲜切工坊-新/中山脆皖/01010000415000000.mp4_33603_取走目标的过线帧.jpg,1,中山脆皖

==> stall.txt <==
sku_自选-田园时蔬、锅底鲜汤-新/云南红香麦
sku_自选-田园时蔬、锅底鲜汤-新/冬瓜

==> train_templates.txt <==
/media/hao/image/Haidilao/dapaidang/image/classify/images/2025-12-24/sku_train_李玉乐/sku_自选-田园时蔬、锅底鲜汤-新/红酸汤/01010000419000000_44910_53.jpg,144,红酸汤锅
/media/hao/image/Haidilao/dapaidang/image/classify/images/2025-12-24/sku_train_李玉乐/sku_自选-锅底鲜汤-新/红酸汤/01000004946000000_41382_16.jpg,144,红酸汤锅

==> train.txt <==
/media/hao/image/Haidilao/dapaidang/image/classify/images/2025-12-24/sku_train_李玉乐/sku_自选-锅底鲜汤-新/红酸汤/01000004946000000_41382_51.jpg,144,红酸汤锅
/media/hao/image/Haidilao/dapaidang/image/classify/images/2025-12-24/sku_train_孙昊/自选-甄选海鲜2-新/淡菜/01010000462000000_59280_55.jpg,63,淡菜

==> val.txt <==
/media/hao/image/Haidilao/dapaidang/image/classify/images/2025-12-24-all-stall-result3/自选-锅底鲜汤-新/花椒鸡锅/01010000411000000.mp4_10456_取走目标的过线帧.jpg,111,花椒鸡锅
/media/hao/image/Haidilao/dapaidang/image/classify/images/2025-12-24-all-stall-result3/自选-田园时蔬-新/脆脆毛肚/01010000413000000.mp4_60963_放回目标的静止帧.jpg,26,脆脆毛肚
```

## 常见问题
- **为什么有 `待定N`？**  
  目录名是纯数字时，脚本会按目录路径生成 `待定N`，用于区分不同数字目录。
- **为什么 `train.txt` / `val.txt` 是软链接？**  
  便于训练脚本自动发现 `train.txt` / `val.txt`，同时保留真实文件名 `output_train.txt` / `output_test.txt`。
