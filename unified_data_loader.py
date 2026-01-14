#!/usr/bin/env python3
# encoding: utf-8
'''
@author: 孙昊
@contact: smartadpole@163.com
@file: unified_data_loader.py
@time: 2025/01/14 10:00
@desc: Unified data loader supporting multiple dataset formats (ImageFolder, text list, etc.)
'''

import os
from typing import Optional, Tuple, Callable
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms


class TextListDataset(Dataset):
    """
    Dataset from text file with format: image_path label or image_path,label
    """
    def __init__(self, txt_path: str, image_root: str, transform: Optional[Callable] = None):
        self.image_root = image_root
        self.transform = transform
        self.samples = []
        
        with open(txt_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                # Support both space and comma separators
                if ',' in line:
                    parts = line.split(',')
                else:
                    parts = line.split(' ')

                img_path, label = parts[0].strip(), int(parts[1].strip())
                self.samples.append((img_path, label))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        # Handle both absolute and relative paths
        if os.path.isabs(img_path):
            full_path = img_path
        else:
            full_path = os.path.join(self.image_root, img_path)
        img = Image.open(full_path).convert('RGB')
        
        if self.transform:
            img = self.transform(img)
        
        return img, label


def detect_dataset_format(data_root: str, train_path: Optional[str] = None, 
                         val_path: Optional[str] = None) -> str:
    """
    Auto-detect dataset format.
    
    Returns:
        'imagefolder': ImageFolder format (train/val subdirectories)
        'textlist': Text list format (train.txt/val.txt files)
        'mixed': Mixed format (explicit paths provided)
    """
    # Check if explicit paths are provided
    if train_path and val_path:
        if os.path.isfile(train_path) and os.path.isfile(val_path):
            return 'textlist'
        elif os.path.isdir(train_path) and os.path.isdir(val_path):
            return 'imagefolder'
        else:
            raise ValueError(f"Cannot determine format for train_path={train_path}, val_path={val_path}")
    
    # Auto-detect from data_root
    train_dir = os.path.join(data_root, 'train')
    val_dir = os.path.join(data_root, 'val')
    train_txt = os.path.join(data_root, 'train.txt')
    val_txt = os.path.join(data_root, 'val.txt')
    
    # Check ImageFolder format
    if os.path.isdir(train_dir) and os.path.isdir(val_dir):
        return 'imagefolder'
    
    # Check text list format
    if os.path.isfile(train_txt) and os.path.isfile(val_txt):
        return 'textlist'
    
    # Try alternative names
    test_txt = os.path.join(data_root, 'test.txt')
    if os.path.isfile(train_txt) and os.path.isfile(test_txt):
        return 'textlist'
    
    raise ValueError(
        f"Cannot auto-detect dataset format from {data_root}. "
        f"Expected either:\n"
        f"  1. ImageFolder: {train_dir}/ and {val_dir}/ directories\n"
        f"  2. Text list: {train_txt} and {val_txt} files"
    )


def load_unified_dataset(
    data_root: str,
    train_path: Optional[str] = None,
    val_path: Optional[str] = None,
    train_transform: Optional[Callable] = None,
    val_transform: Optional[Callable] = None,
    batch_size: int = 32,
    num_workers: int = 4,
    pin_memory: bool = True,
    shuffle_train: bool = True,
    **dataloader_kwargs
) -> Tuple[Dataset, DataLoader, Dataset, DataLoader]:
    """
    Unified dataset loader supporting multiple formats.
    
    Args:
        data_root: Root directory of dataset (for auto-detection and image paths)
        train_path: Explicit path to training data (file or directory)
        val_path: Explicit path to validation data (file or directory)
        train_transform: Transform for training data
        val_transform: Transform for validation data
        batch_size: Batch size for DataLoader
        num_workers: Number of worker processes
        pin_memory: Whether to pin memory for faster GPU transfer
        shuffle_train: Whether to shuffle training data
        **dataloader_kwargs: Additional arguments for DataLoader
    
    Returns:
        Tuple of (train_dataset, train_loader, val_dataset, val_loader)
    """
    # Auto-detect format if paths not explicitly provided
    if train_path and val_path:
        fmt = detect_dataset_format(data_root, train_path, val_path)
    else:
        fmt = detect_dataset_format(data_root)
    
    # Determine actual paths
    if fmt == 'imagefolder':
        if train_path and os.path.isdir(train_path):
            train_dir = train_path
        else:
            train_dir = os.path.join(data_root, 'train')
        
        if val_path and os.path.isdir(val_path):
            val_dir = val_path
        else:
            val_dir = os.path.join(data_root, 'val')
        
        if not os.path.isdir(train_dir):
            raise ValueError(f"Training directory not found: {train_dir}")
        if not os.path.isdir(val_dir):
            raise ValueError(f"Validation directory not found: {val_dir}")
        
        # Load ImageFolder datasets
        train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
        val_dataset = datasets.ImageFolder(val_dir, transform=val_transform)
        
    elif fmt == 'textlist':
        # Determine text file paths
        if train_path and os.path.isfile(train_path):
            train_txt = train_path
        else:
            train_txt = os.path.join(data_root, 'train.txt')
            if not os.path.isfile(train_txt):
                test_txt = os.path.join(data_root, 'test.txt')
                if os.path.isfile(test_txt):
                    train_txt = test_txt
        
        if val_path and os.path.isfile(val_path):
            val_txt = val_path
        else:
            val_txt = os.path.join(data_root, 'val.txt')
            if not os.path.isfile(val_txt):
                test_txt = os.path.join(data_root, 'test.txt')
                if os.path.isfile(test_txt):
                    val_txt = test_txt
        
        if not os.path.isfile(train_txt):
            raise ValueError(f"Training text file not found: {train_txt}")
        if not os.path.isfile(val_txt):
            raise ValueError(f"Validation text file not found: {val_txt}")
        
        # Load text list datasets (use data_root as image root for relative paths)
        train_dataset = TextListDataset(train_txt, data_root, transform=train_transform)
        val_dataset = TextListDataset(val_txt, data_root, transform=val_transform)
    
    else:
        raise ValueError(f"Unsupported dataset format: {fmt}")
    
    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train,
        num_workers=num_workers,
        pin_memory=pin_memory,
        **dataloader_kwargs
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        **dataloader_kwargs
    )
    
    return train_dataset, train_loader, val_dataset, val_loader
