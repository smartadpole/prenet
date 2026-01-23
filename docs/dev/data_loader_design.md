# Unified Data Loader Design

## Overview

The unified data loader (`data/unified_data_loader.py`) provides a flexible interface for loading datasets in multiple formats, supporting both ImageFolder-style directory structures and text-based list files. This design enables seamless integration with various dataset organizations while maintaining a consistent API.

## Design Rationale

### Problem Context

Different datasets come in different formats:
- **ImageFolder**: Organized as `train/class1/`, `train/class2/`, etc.
- **Text List**: Simple text files with `image_path label` format
- **Mixed**: Explicit paths provided via command-line arguments

A unified loader eliminates the need for format-specific code in training scripts.

### Core Design Principles

1. **Auto-Detection**: Automatically detect dataset format from directory structure or file presence
2. **Flexibility**: Support both explicit paths and auto-discovery
3. **Compatibility**: Maintain compatibility with torchvision's ImageFolder API
4. **Error Handling**: Provide clear error messages when format cannot be determined

## Architecture

### Components

#### 1. TextListDataset

**Purpose**: Load datasets from text files with format `image_path label` or `image_path,label`

**Key Features**:
- Supports both space and comma separators
- Handles absolute and relative image paths
- Automatic RGB conversion for all images

**Implementation**:
```python
class TextListDataset(Dataset):
    def __init__(self, txt_path, image_root, transform=None):
        # Parse text file, store (img_path, label) pairs
        # image_root used for resolving relative paths
    
    def __getitem__(self, idx):
        # Load image, apply transform, return (image, label)
```

**Path Resolution Logic**:
- If `img_path` is absolute → use directly
- If `img_path` is relative → join with `image_root`

#### 2. Format Detection

**Function**: `detect_dataset_format()`

**Detection Priority**:
1. **Explicit Paths**: If `train_path` and `val_path` provided, check if files or directories
2. **ImageFolder Format**: Check for `train/` and `val/` subdirectories in `data_root`
3. **Text List Format**: Check for `train.txt` and `val.txt` (or `test.txt`) in `data_root`

**Return Values**:
- `'imagefolder'`: Directory-based structure
- `'textlist'`: Text file-based structure
- Raises `ValueError` if format cannot be determined

#### 3. Unified Loader

**Function**: `load_unified_dataset()`

**Workflow**:
```
1. Detect format (auto or explicit)
2. Resolve actual paths (explicit or auto-discovered)
3. Create appropriate Dataset class
4. Wrap in DataLoader with specified parameters
5. Return (train_dataset, train_loader, val_dataset, val_loader)
```

## Supported Formats

### Format 1: ImageFolder

**Directory Structure**:
```
data_root/
├── train/
│   ├── class_001/
│   │   ├── img1.jpg
│   │   └── img2.jpg
│   └── class_002/
│       ├── img1.jpg
│       └── img2.jpg
└── val/
    ├── class_001/
    └── class_002/
```

**Auto-Detection**: Looks for `data_root/train/` and `data_root/val/` directories

**Explicit Paths**: Can provide `train_path` and `val_path` as directory paths

**Implementation**: Uses `torchvision.datasets.ImageFolder`

### Format 2: Text List

**File Structure**:
```
data_root/
├── train.txt
└── val.txt
```

**File Format** (space or comma separated):
```
class_001/img1.jpg 0
class_001/img2.jpg 0
class_002/img1.jpg 1
class_002/img2.jpg 1
```

**Alternative Format** (comma separated):
```
class_001/img1.jpg,0
class_001/img2.jpg,0
```

**Auto-Detection**: Looks for `data_root/train.txt` and `data_root/val.txt` (or `test.txt`)

**Explicit Paths**: Can provide `train_path` and `val_path` as file paths

**Implementation**: Uses custom `TextListDataset` class

## API Reference

### load_unified_dataset()

**Signature**:
```python
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
) -> Tuple[Dataset, DataLoader, Dataset, DataLoader]
```

**Parameters**:
- `data_root`: Root directory for dataset (required for auto-detection and relative path resolution)
- `train_path`: Explicit path to training data (file or directory, optional)
- `val_path`: Explicit path to validation data (file or directory, optional)
- `train_transform`: Transform pipeline for training data
- `val_transform`: Transform pipeline for validation data
- `batch_size`: Batch size for DataLoader (default: 32)
- `num_workers`: Number of worker processes (default: 4)
- `pin_memory`: Pin memory for faster GPU transfer (default: True)
- `shuffle_train`: Shuffle training data (default: True)
- `**dataloader_kwargs`: Additional arguments passed to DataLoader

**Returns**:
- `train_dataset`: Training dataset object
- `train_loader`: Training DataLoader
- `val_dataset`: Validation dataset object
- `val_loader`: Validation DataLoader

## Usage Examples

### Example 1: Auto-Detection (ImageFolder)

```python
from data.unified_data_loader import load_unified_dataset

train_set, train_loader, val_set, val_loader = load_unified_dataset(
    data_root="/path/to/dataset",
    train_transform=train_tfm,
    val_transform=val_tfm,
    batch_size=64
)
```

### Example 2: Auto-Detection (Text List)

```python
train_set, train_loader, val_set, val_loader = load_unified_dataset(
    data_root="/path/to/dataset",  # Contains train.txt and val.txt
    train_transform=train_tfm,
    val_transform=val_tfm,
    batch_size=64
)
```

### Example 3: Explicit Paths

```python
train_set, train_loader, val_set, val_loader = load_unified_dataset(
    data_root="/path/to/dataset",
    train_path="/path/to/train.txt",
    val_path="/path/to/val.txt",
    train_transform=train_tfm,
    val_transform=val_tfm,
    batch_size=64
)
```

## Design Decisions

### Why Support Multiple Formats?

1. **Dataset Diversity**: Real-world datasets come in various formats
2. **Legacy Compatibility**: Existing datasets may use text list format
3. **Flexibility**: Users can choose format based on their workflow

### Why Auto-Detection?

1. **User Convenience**: Reduces configuration burden
2. **Error Prevention**: Clear error messages when format unclear
3. **Backward Compatibility**: Works with existing dataset structures

### Why TextListDataset Instead of Standard ImageFolder?

1. **Flexibility**: Text files allow custom image organization
2. **Metadata**: Can include additional information in text format
3. **Cross-Platform**: Text files are easier to generate/modify

## Error Handling

### Format Detection Errors

**Scenario**: Cannot determine dataset format

**Error Message**:
```
ValueError: Cannot auto-detect dataset format from {data_root}.
Expected either:
  1. ImageFolder: {train_dir}/ and {val_dir}/ directories
  2. Text list: {train_txt} and {val_txt} files
```

### Missing File/Directory Errors

**Scenario**: Required files or directories not found

**Error Messages**:
- `ValueError: Training directory not found: {train_dir}`
- `ValueError: Validation directory not found: {val_dir}`
- `ValueError: Training text file not found: {train_txt}`
- `ValueError: Validation text file not found: {val_txt}`

## Integration with Training Scripts

The unified loader is used in:
- `train_dinov2_arcface_small.py`: Main training script
- `data/data_loader.py`: Legacy wrapper (maintains backward compatibility)

**Integration Pattern**:
```python
from data.unified_data_loader import load_unified_dataset

train_set, train_loader, val_set, val_loader = load_unified_dataset(
    data_root=args.data_root,
    train_path=args.train_path,
    val_path=args.val_path,
    train_transform=build_train_tfm(img_size),
    val_transform=build_val_tfm(img_size),
    batch_size=args.batch_size,
    num_workers=args.num_workers,
    pin_memory=(device == "cuda"),
    shuffle_train=True
)
```

## Future Improvements

1. **Additional Formats**: Support for CSV, JSON, or other structured formats
2. **Streaming**: Support for large datasets that don't fit in memory
3. **Caching**: Cache parsed text files for faster subsequent loads
4. **Validation**: Validate image paths and labels during initialization
5. **Multi-Label Support**: Extend to support multi-label classification

## Related Documentation

- **Training Architecture**: `docs/dev/dinov2_training_architecture.md`
- **Data Augmentation**: `docs/dev/training_data_augmentation.md`
