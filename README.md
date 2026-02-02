# Progressive Region Enhancement Network (PRENet)

Code release for Large Scale Visual Food Recognition

**Version:** 0.1.21

### Introduction
![method](Method.png)

Our Progressive Region Enhancement Network (PRENet) mainly consists of progressive local feature learning and region feature enhancement. The former mainly adopts the progressive training strategy to learn complementary multi-scale finer local features, like different ingredient-relevant information. The region feature enhancement uses self-attention to incorporate richer contexts with multiple scales into local features to enhance the local feature representation. Then we fuse enhanced local features and global ones from global feature learning into the unified one via the concat layer.

During training, after progressively training the networks from different stages, we then train the whole network with the concat part, and further introduce the KL-divergence to increase the difference between stages for capturing more detailed features. For the inference, considering the complementary output from each stage and the concatenated features, we combine the prediction results from them for final food classification.
 
### Requirement
 
- python 3.6

- PyTorch >= 1.3.1

- torchvision >= 0.4.2

- PIL (Pillow)

- Numpy 

- pandas

- tqdm

- matplotlib

- dropblock

**Quick Installation:**

```bash
pip install -r requirements.txt
```

### Data preparation


1. For dataset production using the provided scripts (including `gen_all_dataset.sh`) and a detailed description of generated files, see:
    - [docs/user/dataset_preparation.md](docs/user/dataset_preparation.md)


2. Download the food datasets. The file structure should look like:
    ```
    dataset
    ├── class_001
    |      ├── 1.jpg
    |      ├── 2.jpg
    |      └── ...
    ├── class_002
    |      ├── 1.jpg
    |      ├── 2.jpg
    |      └── ...
    │── ...
    ```

3. Prepare the training and testing list files. You have two options:

   **Option A:** Place `train.txt` and `val.txt` in the dataset root directory. The data loader will automatically discover them.
   
   **Option B:** Provide explicit paths using `--train_path` and `--test_path` arguments.

   The list files should contain one entry per line, with format:
   - Space-separated: `image_path label`
   - Comma-separated: `image_path,label`
   
   Example:
   ```
   class_001/1.jpg 0
   class_001/2.jpg 0
   class_002/1.jpg 1
   ``` 


### Training

1. To train a `PRENet` on food datasets from scratch, run:

   **If using automatic file discovery (train.txt/val.txt in dataset root):**
   ```bash
   python main.py --dataset <food_dataset> --image_path <data_path> --weight_path <pretrained_model>
   ```

   **If using explicit paths:**
   ```bash
   python main.py --dataset <food_dataset> --image_path <data_path> --train_path <train_path> --test_path <test_path> --weight_path <pretrained_model>
   ```

   **Parameters:**
   - `--dataset`: Dataset type (`food2k`, `food101`, `food500`, or `other`). Default: `other`
   - `--image_path`: (Required) Path to dataset root directory
   - `--train_path`: (Optional) Path to training list file. If not provided, will look for `train.txt` in `image_path`
   - `--test_path`: (Optional) Path to testing list file. If not provided, will look for `val.txt` in `image_path`
   - `--weight_path`: (Required) Path to pretrained model weights
   - `--batchsize`: Batch size (default: 2)
   - `--learning_rate`: Initial learning rate (default: 1e-4)
   - `--epoch`: Number of training epochs (default: 200)

### Inference

1. Download the pretrained model on Food2k from [google](https://drive.google.com/file/d/1gA_abY0d_0B6jXpeXNgCKBbSzc8iEHxU/view?usp=sharing)/[baidu](https://pan.baidu.com/s/1HMvBf0F-FpMIMPtuQtUE8Q)(Code: o0nj)

2. To evaluate a pre-trained `PRENet` on food datasets, run:

   **If using automatic file discovery:**
   ```bash
   python main.py --dataset <food_dataset> --image_path <data_path> --weight_path <pretrained_model> --test --use_checkpoint --checkpoint <checkpoint_path>
   ```

   **If using explicit paths:**
   ```bash
   python main.py --dataset <food_dataset> --image_path <data_path> --train_path <train_path> --test_path <test_path> --weight_path <pretrained_model> --test --use_checkpoint --checkpoint <checkpoint_path>
   ```

### Other pretrained model on Food2K

|  CNN   | link  |
|  ----  | ----  |
| vgg16  | [google](https://drive.google.com/file/d/1r4CQEfCkwLSKz5QdZJGABldercUo5BtF/view?usp=sharing)/[baidu](https://pan.baidu.com/s/1-nI6fodmmzqz9OVqh0yvUw)(Code: puuy)|
| resnet50  | [google](https://drive.google.com/file/d/1h87m392fJIxrADTe8GMH7pibP0rjWu-k/view?usp=sharing)/[baidu](https://pan.baidu.com/s/1WY7VsCBTJt2mL9n3Gdl8Mg)(Code: 5eay) |
| resnet101  | [google](https://drive.google.com/file/d/1_xM2qv1NIjev8voYjXLhfnxDzvFNB85q/view?usp=sharing)/[baidu](https://pan.baidu.com/s/1mEO7KyJFHrkpB5G0Aj6oWw)(Code: yv1o) |
| resnet152  | [google](https://drive.google.com/file/d/1YG_gW6NftjX06-i3bCCYQhlnDo2mUoLn/view?usp=sharing)/[baidu](https://pan.baidu.com/s/1-3LikXkDEvbxQur6n-FUJw)(Code: 22zw) |
| densenet161  | [google](https://drive.google.com/file/d/17PAUHmo1vIM9b4SlbpnLnwp1a5MH9Vem/view?usp=sharing)/[baidu](https://pan.baidu.com/s/1UllqjTJMAQEnGFVgzf6-nQ)(Code: bew5) |
| inception_resnet_v2  | [google](https://drive.google.com/file/d/16PuZRuUB-YFKZT8JWycaay3JdfTlCoVK/view?usp=sharing)/[baidu](https://pan.baidu.com/s/1_974E4eZRzKubemLIQlOHA)(Code: xa8r) |
| senet154  | [google](https://drive.google.com/file/d/1FGs7gH1fYybr3sKB6q4lRl36bX5wiLXw/view?usp=sharing)/[baidu](https://pan.baidu.com/s/1tHpFFSm2AySRjDZ4BTtboQ)(Code: kwzf) |

---

## DINOv2 ArcFace Model Tools

This project also includes tools for training and evaluating DINOv2-based ArcFace classification models.

### DINOv2 Training

To train a DINOv2 ArcFace model:

```bash
python train_dinov2_arcface_small.py \    
    --data_root <files/v1.0.5>   --output_dir <output mdoel dir> \
    -lr 0.0002 --epoch 110 -b 48 --num_classes <class_num + 1> \
    [OPTIONS]
```

**Required Parameters:**
- `--data_root`: Root directory of the dataset
- `--num_classes`: Number of classification classes (required)

**Optional Parameters:**
- `--train_path`: Path to training data file or directory (default: auto-detect `train.txt` in `data_root`)
- `--val_path`: Path to validation data file or directory (default: auto-detect `val.txt` in `data_root`)
- `--backbone`: DINOv2 backbone model (default: `dinov2_vitb14`)
- `--img_size`: Input image size (default: 128)
- `--embed_dim`: Embedding dimension (default: 256)
- `--epochs`: Total training epochs (default: 20)
- `--batch_size`, `-b`: Batch size (default: 64)
- `--num_workers`: Number of data loading workers (default: 6)
- `--lr_head`, `-lr`: Learning rate for head layers (default: 3e-4)
- `--lr_backbone`: Learning rate for backbone layers (default: 1e-5)
- `--weight_decay`: Weight decay (default: 0.05)
- `--arc_s`: ArcFace scale parameter (default: 32.0)
- `--arc_m`: ArcFace margin parameter (default: 0.30)
- `--stage1_epochs`: Epochs to freeze backbone (default: 12)
- `--unfreeze_blocks`: Number of last blocks to unfreeze in stage 2 (default: 1)
- `--device`: Device to use (`cuda` or `cpu`, default: `cuda`)
- `--output_dir`: Output directory for saved models (default: `output_dinov2_arcface_small`)

The training process uses a two-stage approach:
- **Stage 1**: Freezes the backbone and trains only the projection head and ArcFace head
- **Stage 2**: Unfreezes the last N blocks of the backbone for fine-tuning

### DINOv2 Classification Testing

To classify images using a trained DINOv2 ArcFace model:

```bash
python tools/test_dinov2_classification.py \
    --model_path <checkpoint_path> \
    --image_dir <image_directory> \
    [OPTIONS]
```

**Required Parameters:**
- `--model_path`: Path to model checkpoint (.pt file)
- `--image_dir`: Directory containing images (supports nested directories)

**Optional Parameters:**
- `--output_dir`: Output directory for results (default: `test_output`)
- `--device`: Device to use (`cuda` or `cpu`, default: `cuda`)
- `--batch_size`, `-b`: Batch size for inference (default: 32)
- `--max_images_per_class`: Maximum images to show per class in visualization (default: 20)
- `--label`: Path to label file (.txt) with class names (optional)

**Output:**
- CSV file with classification results (`classification_results.csv`)
- Visualization images grouped by predicted category (one image per class)

### DINOv2 Accuracy Evaluation

To evaluate model accuracy with ground truth labels:

```bash
python tools/eval.py \
    --model_path <checkpoint_path> \
    --test_file <test_file> \
    [OPTIONS]
```

**Required Parameters:**
- `--model_path`: Path to model checkpoint (.pt file)
- `--test_file`: Path to test file (.txt) with ground truth labels

**Test File Format:**
The test file should contain three columns per line (space or tab separated):
```
<absolute_image_path> <label_id> <class_name>
```

Example:
```
/path/to/image1.jpg 0 类别A
/path/to/image2.jpg 1 类别B
/path/to/image3.jpg 0 类别A
```

**Optional Parameters:**
- `--output_dir`: Output directory for results (default: `eval_output`)
- `--device`: Device to use (`cuda` or `cpu`, default: `cuda`)
- `--batch_size`, `-b`: Batch size for inference (default: 32)

**Output:**
- Text file with accuracy metrics (`evaluation_results.txt`)
- Visualization images per class showing:
  - Top 10 correct predictions (by confidence)
  - All wrong predictions with true and predicted labels
  - Accuracy, wrong count, and total count displayed on each image

**Metrics Calculated:**
- Overall accuracy (with correct/wrong/total counts)
- Per-class accuracy (with correct/wrong/total counts for each class)

### DINOv2 Open-Set Recognition Evaluation

To evaluate model with open-set recognition using offline feature gallery:

```bash
python tools/eval_open.py \
    --model_path <checkpoint_path> \
    --template_path <template_path> \
    --test_file <test_file> \
    [OPTIONS]
```

**Required Parameters:**
- `--model_path`: Path to model checkpoint (.pt file)
- `--template_path`: Path to template file (.txt/.csv) or directory for building feature gallery
- `--test_file`: Path to test file (.txt) with ground truth labels

**File Format:**
Template and test files should contain three columns per line (space/tab/comma separated):
```
<absolute_image_path> <label_id> <class_name>
```

**Template Directory Format:**
```
<template_root>/
├── <label_name_1>/
│   ├── img1.jpg
│   └── ...
└── <label_name_2>/
    └── ...
```

**Optional Parameters:**
- `--output_dir`: Output directory for results (default: `eval_output`)
- `--device`: Device to use (`cuda` or `cpu`, default: `cuda`)
- `--top_k`: Number of nearest neighbors for voting (default: 5)
- `--threshold`: Absolute similarity threshold for open-set rejection (default: 0.6)
- `--margin_threshold`: Margin threshold between top-1 and top-2 scores (default: 0.1)
- `--outlier_threshold`: Outlier removal threshold in standard deviations (default: 2.0)
- `--batch_size`, `-b`: Batch size for inference (default: 32)
- `--save_vis`: Save visualization images (default: disabled)

**Features:**
- Offline feature gallery construction from template images
- Top-K nearest neighbor search with weighted voting
- Class imbalance handling using size normalization
- Dual-threshold open-set rejection (absolute + relative)
- Unknown class detection with error counting in metrics
- Template gallery cache stored next to the template directory as `gallery_cache_<model_version>`

**Output:**
- Text file with accuracy metrics (`evaluation_results.txt`)
- Optional visualization images per class (enabled with `--save_vis`) showing:
  - Top 10 correct predictions (by confidence)
  - All wrong predictions with true and predicted labels
  - Unknown predictions marked as `Unknown/Rejected`

**Key Differences from Closed-Set Evaluation:**
- Uses metric learning instead of classification head
- Supports dynamic class addition without retraining
- Can identify "unknown" classes not in the gallery
- Handles class imbalance through weighted voting

### CSV Batch Inference with Bbox Cropping

To perform batch inference on images with bbox cropping from CSV file:

```bash
python tools/test_full.py \
    --input_csv <input_csv_path> \
    --model_path <checkpoint_path> \
    [OPTIONS]
```

**Required Parameters:**
- `--input_csv`: Path to input CSV file containing image paths and bbox information
- `--model_path`: Path to model checkpoint (.pt file)

**Optional Parameters:**
- `--suffix`: Output CSV file suffix name (default: `label`). Output file will be `{input_csv_basename}_{suffix}.csv`
- `--label_file`: Path to label file (.txt) with class names (optional)
- `--template_path`: Template file or directory for open-set retrieval (optional; when set, enable open-set mode)
- `--base_dir`: Base directory for resolving relative image paths (default: directory of input_csv)
- `--device`: Device to use (`cuda` or `cpu`, default: `cuda`)
- `--batch_size`, `-b`: Batch size for inference (default: 32)
- `--crop_dir`: Temporary directory for saving cropped images (default: None; when not specified, uses subdirectory `temp_cropped` under the input CSV directory and removes it after processing; when specified, it is also removed after processing)
- `--save_dir`: Directory to save cropped images organized by category (optional)
- `--visualize`: Enable visualization of images with bbox, label and confidence (default: False)
- `--vis_output_dir`: Output directory for visualized images (default: `visualizations`, only used if `--visualize` is set)
- `--top_k`: Open-set only, number of nearest neighbors for voting (default: 5)
- `--threshold`: Open-set only, absolute similarity threshold for rejection (default: 0.6)
- `--margin_threshold`: Open-set only, margin threshold between top-1 and top-2 scores (default: 0.1)
- `--outlier_threshold`: Open-set only, outlier removal threshold (default: 2.0)
- `--allow_unknown`: Open-set only, allow `Unknown/Rejected` output (default: disabled)

**Input CSV Format:**
The CSV file should contain the following columns:
- `take_first_image_name`, `take_first_bbox`: First frame image path and bbox
- `take_cross_image_name`, `take_cross_bbox`: Crossing line frame image path and bbox
- `return_image_name`, `return_bbox`: Return crossing frame image path and bbox
- `return_static_image_name`, `return_static_bbox`: Return static frame image path and bbox

Bbox format: normalized coordinates "cx cy w h" (space-separated, center point + width/height, values in [0, 1])

**Output:**
- CSV file with 8 additional columns containing classification results:
  - `take_first_image_label`, `take_first_image_confidence`
  - `take_cross_image_label`, `take_cross_image_confidence`
  - `return_image_label`, `return_image_confidence`
  - `return_static_image_label`, `return_static_image_confidence`
- If `--template_path` is set, the tool uses open-set retrieval with template gallery
  - If `--label_file` is provided and valid, retrieval is restricted to those classes only
  - If `--allow_unknown` is enabled, rejected samples are labeled as `Unknown/Rejected`
- If `--save_dir` is specified, cropped images are saved organized by category (subdirectories named by class name)
- If `--visualize` is enabled, visualization images are saved to the specified output directory

**Visualization:**
When `--visualize` is enabled, the tool generates visualization images showing:
- Original image with red bounding box (bbox)
- Label text with confidence score displayed near the bbox
- Format: `class_name: confidence` (confidence with 3 decimal places)

### Batch Script for Count Results CSV

The batch script scans the working directory for `*count_results.csv` files and runs `tools/test_full.py` for each file.

**Example (open-set with unknown output and custom save root):**
```bash
cd /home/user/data/video/2026-01-01-all-stall-convert-result2-big
bash /home/user/sunhao/CODE/prenet/scripts/batch_test_full_140.sh \
  --template_path /home/user/data/image/classify/files/train/v1.0.5/sku \
  --allow_unknown \
  --save_root /home/user/sunhao/Result/20260101_openset_unknown
```

**Notes:**
- Working directory matters: the script only searches under the current directory.
- `--template_path` enables open-set retrieval automatically.
- `--save_root` controls whether cropped images are saved; if empty, no images are saved.

### Template Library Generation

To generate template library using fastdup clustering-based sampling for open-set recognition:

```bash
python tools/gen_sku.py --input <path_or_file> [OPTIONS]
```

**Input (required):**
- **Directory**: Recursive scan; leaf dirs that contain images are treated as categories (supports nested structure).
- **File**: File-list path. Format: 3 columns (path, id, class_name), tab/space/comma separated, same as `eval.load_test_file`. Runs fastdup on involved folders once, then per-label cluster sampling.

**Optional Parameters:**
- `--output`: Output .txt file path for template list (default: `<input>_templates.txt` for directory, or same as input for file)
- `--method`: Sampling method - `center` for cluster center sampling, `hybrid` for center + edge sampling (default: `center`)
- `--num_templates`: Number of templates to generate per category (default: 20)
- `--edge_ratio`: Ratio of edge samples for hybrid method, 0.0-1.0 (default: 0.2, means 20%)
- `--num_em_iter`: Number of EM iterations for KMeans (default: 30, optimized for complex environments)

**Output:**
- Single .txt file with one line per template: `path,label_id,class_name`. Temporary fastdup work dirs are created under the same directory as the output file (`work_dirs/`).

**Sampling Methods:**
- **Center Sampling (`--method center`)**: Clusters images into K clusters and extracts the image closest to each cluster center.
- **Hybrid Sampling (`--method hybrid`)**: Combines center samples with edge samples (max distance in scattered clusters) for robustness.

**Examples:**
```bash
# Recursive directory mode
python tools/gen_sku.py --input ./my_dataset --method center --num_templates 15

# File-list mode
python tools/gen_sku.py --input /path/to/list.txt --method hybrid --num_templates 20 --edge_ratio 0.2
```

## Contact
If you find this repo useful to your project, please consider to cite it with following bib:
```
@article{min2021large,
  title={Large scale visual food recognition},
  author={Min, Weiqing and Wang, Zhiling and Liu, Yuxin and Luo, Mengjiang and Kang, Liping and Wei, Xiaoming and Wei, Xiaolin and Jiang, Shuqiang},
  journal={arXiv preprint arXiv:2103.16107},
  year={2021}
}
```
