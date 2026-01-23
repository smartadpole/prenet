# DINOv2 ArcFace Training Architecture

## Overview

This document describes the architecture and design decisions for the DINOv2-based ArcFace classification training system implemented in `train_dinov2_arcface_small.py`. The system is designed for fine-grained visual classification tasks, particularly optimized for scenarios where subtle texture and color differences are critical discriminative features.

## System Architecture

### High-Level Flow

```
Input Images
    ↓
Data Augmentation (Fine-grained Optimized)
    ↓
DINOv2 Backbone (ViT-based)
    ↓
Patch Token Mean Pooling
    ↓
Projection Head (2-layer MLP)
    ↓
L2 Normalization
    ↓
ArcFace Classification Head
    ↓
Cross-Entropy Loss
    ↓
Two-Stage Training (Freeze → Unfreeze)
```

## Core Components

### 1. Data Augmentation Pipeline

**Location**: `build_train_tfm()` function in `train_dinov2_arcface_small.py`

**Design Philosophy**: Conservative augmentation strategy optimized for fine-grained classification tasks. The pipeline preserves critical discriminative features (texture patterns, color tones) while simulating realistic environmental variations.

**Key Components**:
- **Geometric Transforms**: Conservative rotation, translation, scaling (no shear)
- **Color Augmentation**: Minimal hue shift (0.01), moderate brightness/contrast/saturation
- **Environmental Simulation**: Light blur for smoke/haze, reduced JPEG compression artifacts
- **Occlusion**: Single RandomErasing with reduced scale to protect key textures

**Detailed Documentation**: See `docs/dev/training_data_augmentation.md` for complete parameter specifications and design rationale.

### 2. DINOv2 Embedder

**Class**: `DinoV2Embedder`

**Architecture**:
```
DINOv2 Backbone (ViT)
    ↓
forward_features() → patch tokens [B, N, D]
    ↓
Mean Pooling → [B, D]
    ↓
Projection Head (Linear → GELU → Linear)
    ↓
L2 Normalization → [B, embed_dim]
```

**Key Design Decisions**:

1. **Backbone Selection**: Uses pre-trained DINOv2 ViT models (default: `dinov2_vitb14`)
   - **Rationale**: DINOv2 provides strong self-supervised features that transfer well to downstream tasks
   - **Implementation**: Loads from torch.hub with local cache support

2. **Feature Extraction**: Mean pooling over patch tokens
   - **Rationale**: Aggregates spatial information into a single global descriptor
   - **Alternative Considered**: CLS token, but mean pooling provides better spatial aggregation

3. **Projection Head**: 2-layer MLP with GELU activation
   - **Structure**: `Linear(in_dim, embed_dim) → GELU → Linear(embed_dim, embed_dim)`
   - **Rationale**: Non-linear projection helps adapt pre-trained features to task-specific embedding space

4. **Normalization**: L2 normalization of final embeddings
   - **Rationale**: Required for ArcFace loss computation (cosine similarity)

**Parameters**:
- `backbone_name`: DINOv2 model variant (default: "dinov2_vitb14")
- `embed_dim`: Output embedding dimension (default: 256)
- `train`: Whether to load pre-trained weights (default: True)

### 3. ArcFace Classification Head

**Class**: `ArcFaceHead`

**Architecture**:
```
Normalized Embeddings [B, embed_dim]
    ↓
Learnable Weight Matrix W [num_classes, embed_dim]
    ↓
Cosine Similarity (L2-normalized)
    ↓
ArcFace Margin Application
    ↓
Scaled Logits [B, num_classes]
```

**ArcFace Loss Formula**:
```
For ground truth class y:
  logit_y = s * cos(θ_y + m)
  logit_i = s * cos(θ_i)  (for i ≠ y)

where:
  θ_y = arccos(W_y · z / ||W_y|| ||z||)
  s = scale parameter (default: 32.0)
  m = margin parameter (default: 0.30)
```

**Key Design Decisions**:

1. **Margin-based Loss**: Uses ArcFace margin to increase inter-class separation
   - **Rationale**: For fine-grained classification, margin helps separate visually similar classes
   - **Margin Value**: 0.30 (standard for face recognition, adapted for fine-grained tasks)

2. **Scale Parameter**: s = 32.0
   - **Rationale**: Controls the magnitude of logits, prevents gradient vanishing

3. **Weight Initialization**: Xavier uniform initialization
   - **Rationale**: Ensures proper gradient flow at initialization

**Parameters**:
- `embed_dim`: Input embedding dimension
- `num_classes`: Number of classification classes
- `s`: Scale parameter (default: 32.0)
- `m`: Margin parameter (default: 0.30)

### 4. Two-Stage Training Strategy

**Design Rationale**: 
- **Stage 1**: Freeze backbone, train only projection head and classification head
  - Allows task-specific adaptation without disrupting pre-trained features
  - Faster training, lower memory usage
  - Prevents catastrophic forgetting of pre-trained knowledge

- **Stage 2**: Unfreeze last N blocks of backbone for fine-tuning
  - Enables domain-specific feature refinement
  - Gradual unfreezing prevents overfitting
  - Only unfreezes top layers (most task-relevant)

**Implementation Flow**:

```python
# Stage 1: Freeze backbone
set_requires_grad(backbone, False)
set_requires_grad(proj, True)
set_requires_grad(head, True)

# Train for stage1_epochs epochs...

# Stage 2: Unfreeze last N blocks
unfreeze_last_blocks_dinov2(embedder, n_blocks=unfreeze_blocks)
# Rebuild optimizer with unfrozen parameters
```

**Key Functions**:
- `set_requires_grad()`: Utility to freeze/unfreeze module parameters
- `unfreeze_last_blocks_dinov2()`: Selectively unfreezes last N transformer blocks and normalization layers

**Parameters**:
- `stage1_epochs`: Number of epochs for Stage 1 (default: 12)
- `unfreeze_blocks`: Number of last blocks to unfreeze in Stage 2 (default: 1)

### 5. Optimization Strategy

**Optimizer**: AdamW with separate learning rates

**Learning Rate Schedule**:
- **Warmup**: Linear warmup for first epoch
- **Main Schedule**: Cosine annealing with half-cycle
  ```
  lr(step) = 0.5 * (1 + cos(π * progress))
  where progress = (step - warmup_steps) / (total_steps - warmup_steps)
  ```

**Learning Rate Separation**:
- `lr_backbone`: 1e-5 (lower, for fine-tuning pre-trained weights)
- `lr_head`: 3e-4 (higher, for training new layers)

**Rationale**:
- Lower LR for backbone prevents overfitting and preserves pre-trained features
- Higher LR for head allows faster adaptation to task-specific features
- Cosine annealing provides smooth convergence

**Additional Features**:
- **Mixed Precision Training**: Uses `autocast` and `GradScaler` for FP16 training (when CUDA available)
- **Weight Decay**: 0.05 (standard for transformer fine-tuning)

### 6. Image Size Handling

**Patch Size Compatibility**: DINOv2 ViT models require input size to be divisible by patch size

**Implementation**:
```python
patch_size = embedder.backbone.patch_embed.patch_size[0]
args.img_size = make_divisible(args.img_size, patch_size)
```

**Function**: `make_divisible(value, divisor)`
- Rounds value to nearest multiple of divisor
- Ensures compatibility with ViT patch embedding

**Default**: 128 (adjusted to nearest patch-size multiple if needed)

## Data Flow

### Training Flow

1. **Data Loading**: Uses `unified_data_loader.py` for flexible dataset formats
2. **Augmentation**: Applies fine-grained optimized augmentation pipeline
3. **Forward Pass**:
   ```
   x [B, 3, H, W] → DINOv2 → z [B, embed_dim] → ArcFace → logits [B, num_classes]
   ```
4. **Loss Computation**: Cross-entropy loss on ArcFace logits
5. **Backward Pass**: Gradient computation with mixed precision support
6. **Optimization**: AdamW update with cosine annealing schedule

### Evaluation Flow

1. **No Augmentation**: Uses `build_val_tfm()` (center crop + resize only)
2. **No Margin**: Evaluation uses cosine similarity without margin
   ```python
   logits = head.s * F.linear(z, W)  # no margin at eval
   ```
3. **Accuracy Calculation**: Top-1 accuracy on validation set

## Model Saving

**Saved Components**:
- `embedder.state_dict()`: DINOv2 embedder weights
- `head.state_dict()`: ArcFace head weights
- `classes`: Class name mapping (if available)
- `args`: Training arguments (for reproducibility)
- `best_train_acc` / `best_val_acc`: Best accuracy achieved
- `epoch`: Training epoch

**Model Files**:
- `best_train_dinov2_arcface_small.pt`: Best training accuracy model
- `best_val_dinov2_arcface_small.pt`: Best validation accuracy model

**Rationale**: Separate models allow choosing based on training vs. validation performance

## Key Design Decisions Summary

### Why DINOv2?
- Strong self-supervised pre-training provides rich visual features
- ViT architecture scales well with data and model size
- Good transfer learning performance on downstream tasks

### Why ArcFace?
- Margin-based loss increases inter-class separation
- Particularly effective for fine-grained classification
- Well-established in face recognition, adapted for general classification

### Why Two-Stage Training?
- Prevents catastrophic forgetting of pre-trained features
- Allows task-specific adaptation before fine-tuning
- More stable training, better convergence

### Why Conservative Augmentation?
- Fine-grained tasks rely on subtle features (texture, color)
- Aggressive augmentation destroys discriminative information
- Balance between robustness and feature preservation

## Configuration Parameters

### Model Parameters
- `backbone`: DINOv2 model variant (default: "dinov2_vitb14")
- `embed_dim`: Embedding dimension (default: 256)
- `num_classes`: Number of classes (required)
- `arc_s`: ArcFace scale (default: 32.0)
- `arc_m`: ArcFace margin (default: 0.30)

### Training Parameters
- `epochs`: Total training epochs (default: 20)
- `stage1_epochs`: Stage 1 epochs (default: 12)
- `unfreeze_blocks`: Blocks to unfreeze in Stage 2 (default: 1)
- `batch_size`: Batch size (default: 64)
- `lr_head`: Head learning rate (default: 3e-4)
- `lr_backbone`: Backbone learning rate (default: 1e-5)
- `weight_decay`: Weight decay (default: 0.05)

### Data Parameters
- `img_size`: Input image size (default: 128, auto-adjusted for patch size)
- `data_root`: Dataset root directory (required)
- `train_path`: Training data path (optional, auto-detects train.txt)
- `val_path`: Validation data path (optional, auto-detects val.txt)

## Performance Considerations

### Memory Optimization
- Mixed precision training (FP16) reduces memory usage
- Gradient checkpointing not used (DINOv2 is relatively memory-efficient)

### Training Speed
- Two-stage training: Stage 1 is faster (frozen backbone)
- Batch size default: 64 (adjustable based on GPU memory)

### Scalability
- Supports various DINOv2 model sizes (small, base, large)
- Embedding dimension adjustable (default: 256)
- Image size adjustable (must be patch-size divisible)

## Future Improvements

1. **Advanced Augmentation**: Consider CutMix for texture-focused tasks
2. **Label Smoothing**: For visually similar categories
3. **Higher Resolution**: Increase img_size for better texture recognition
4. **Adaptive Learning Rates**: Per-layer or per-parameter learning rates
5. **Multi-Scale Training**: Train with multiple image sizes
6. **Ensemble Methods**: Combine multiple model checkpoints

## Related Documentation

- **Data Augmentation**: `docs/dev/training_data_augmentation.md`
- **User Guide**: `docs/user/test_dinov2_classification.md`
- **Testing Tools**: `docs/dev/test_dinov2_classification.md`, `docs/dev/test_full.md`

## References

- DINOv2: [Oquab et al., 2023] "DINOv2: Learning Robust Visual Features without Supervision"
- ArcFace: [Deng et al., 2019] "ArcFace: Additive Angular Margin Loss for Deep Face Recognition"
- Fine-Grained Classification: Literature on texture and color-sensitive classification tasks
