import torch
import PIL
from PIL import Image
import torch.utils.data as data
from torchvision import transforms
import os
from dataset.unified_data_loader import load_unified_dataset

def My_loader(path):
    return PIL.Image.open(path).convert('RGB')

class MyDataset(torch.utils.data.Dataset):
    """
    Legacy dataset class for backward compatibility.
    Now uses unified_data_loader internally, but kept for API compatibility.
    """

    def __init__(self, txt_dir, image_path, transform=None, target_transform=None, loader=My_loader, use_absolute_path=False):
        data_txt = open(txt_dir, 'r')
        imgs = []
        for line in data_txt:
            line = line.strip()
            words = line.split(',') if ',' in line else line.split(' ')
            imgs.append((words[0], int(words[1].strip())))

        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        self.loader = My_loader
        self.image_path = image_path

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        img_name, label = self.imgs[index]
        img = self.loader(os.path.join(self.image_path, img_name))

        if self.transform is not None:
            img = self.transform(img)
        return img, label
        # if the label is the single-label it can be the int
        # if the multilabel can be the list to torch.tensor

def load_data(image_path, train_dir="", test_dir="", batch_size=1):
    """
    Legacy data loading function for backward compatibility.
    Now uses unified_data_loader internally.
    """
    normalize = transforms.Normalize(mean=[0.5457954, 0.44430383, 0.34424934],
                                     std=[0.23273608, 0.24383051, 0.24237761])
    resize_size = (550, 550)
    crop_size = (448, 448)
    train_transforms = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),  # default value is 0.5
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.126, saturation=0.5),
        transforms.Resize(resize_size),
        transforms.RandomCrop(crop_size),
        transforms.ToTensor(),
        normalize
    ])

    # transforms of test dataset
    test_transforms = transforms.Compose([
        transforms.Resize(resize_size),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        normalize
    ])
    
    # Use unified data loader
    # Determine actual paths for backward compatibility
    actual_train_dir = train_dir
    actual_test_dir = test_dir
    
    # Auto-detect if paths not provided
    if not train_dir or not test_dir:
        train_txt = os.path.join(image_path, 'train.txt')
        val_txt = os.path.join(image_path, 'val.txt')
        if os.path.isfile(train_txt) and os.path.isfile(val_txt):
            actual_train_dir = train_txt
            actual_test_dir = val_txt
    
    # Use unified loader
    train_dataset, train_loader, test_dataset, test_loader = load_unified_dataset(
        data_root=image_path,
        train_path=actual_train_dir if actual_train_dir else None,
        val_path=actual_test_dir if actual_test_dir else None,
        train_transform=train_transforms,
        val_transform=test_transforms,
        batch_size=batch_size,
        num_workers=0,
        pin_memory=False,
        shuffle_train=True
    )
    
    # Adjust test batch size to match legacy behavior
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=max(1, batch_size//2),
        shuffle=False,
        num_workers=0
    )
    
    return train_dataset, train_loader, test_dataset, test_loader
