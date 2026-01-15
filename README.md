# Progressive Region Enhancement Network (PRENet)

Code release for Large Scale Visual Food Recognition

**Version:** 0.1.4

### Introduction
![method](Method.png)

Our Progressive Region Enhancement Network (PRENet) mainly consists of progressive local feature learning and region feature enhancement. The former mainly adopts the progressive training strategy to learn complementary multi-scale finer local features, like different ingredient-relevant information. The region feature enhancement uses self-attention to incorporate richer contexts with multiple scales into local features to enhance the local feature representation. Then we fuse enhanced local features and global ones from global feature learning into the unified one via the concat layer.

During training, after progressively training the networks from different stages, we then train the whole network with the concat part, and further introduce the KL-divergence to increase the difference between stages for capturing more detailed features. For the inference, considering the complementary output from each stage and the concatenated features, we combine the prediction results from them for final food classification.
 
### Requirement
 
- python 3.6

- PyTorch >= 1.3.1

- torchvision >= 0.4.2

- PIL

- Numpy 

- dropblock

### Data preparation

1. Download the food datasets. The file structure should look like:
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

2. Prepare the training and testing list files. You have two options:

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

