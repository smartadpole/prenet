# Training Data Augmentation Strategy

## Overview

This document describes the data augmentation strategy used in `train_dinov2_arcface_small.py` for fine-grained visual classification tasks, particularly for meat classification scenarios where subtle texture and color differences are critical.

## Design Rationale

### Problem Context

For fine-grained classification tasks (e.g., distinguishing different types of meat), the model relies on:
- **Subtle texture patterns** (marbling density, muscle fiber distribution)
- **Precise color tones** (fresh red vs. dark red, slight hue variations)
- **Contextual features** (overall distribution pattern on the plate)

Traditional aggressive data augmentation strategies can destroy these critical discriminative features, leading to poor classification performance.

### Core Design Principles

1. **Preserve Color Fidelity**: Color is a critical feature for fine-grained classification. Excessive color jittering (especially hue shifts) can confuse categories that rely on specific color tones.

2. **Protect Texture Details**: Texture patterns are essential for distinguishing similar objects. Excessive blurring or geometric distortion can destroy these patterns.

3. **Maintain Context**: For objects with fixed backgrounds (e.g., plates), preserving the overall context helps the model learn distribution patterns.

4. **Simulate Realistic Variations**: Augmentations should simulate real-world variations (exposure changes, slight smoke/haze) without introducing unrealistic artifacts.

## Implementation Details

### Geometric Transforms

#### RandomResizedCrop
- **Scale**: `(0.8, 1.0)` (converged from `(0.5, 1.2)`)
  - **Rationale**: Preserve more context features. Too small crops (0.5) may only capture a single red patch, losing the "overall distribution pattern" context.
- **Ratio**: `(0.9, 1.1)` (converged from `(0.75, 1.33)`)
  - **Rationale**: Maintain aspect ratio closer to original, avoiding extreme distortions.

#### RandomHorizontalFlip & RandomVerticalFlip
- **Horizontal Flip**: `p=0.5`
- **Vertical Flip**: `p=0.5` (newly added)
  - **Rationale**: For objects without inherent up-down orientation (e.g., meat on plate), vertical flip increases data diversity without introducing unrealistic variations.

#### RandomAffine
- **Degrees**: `15` (reduced from `20`)
- **Translate**: `(0.1, 0.1)` (reduced from `0.15`)
- **Scale**: `(0.9, 1.1)` (converged from `(0.8, 1.2)`)
- **Shear**: `0` (removed, was `10`)
  - **Rationale**: Shear causes non-physical texture stretching. For rigid objects (plates), removing shear prevents misleading training samples.

#### RandomPerspective
- **Distortion Scale**: `0.1` (reduced from `0.3`)
- **Probability**: `0.3` (reduced from `0.5`)
  - **Rationale**: For fixed camera positions or handheld shots, extreme perspective changes (turning round plates into flat ellipses) are unrealistic and add unnecessary learning burden.

### Color & Lighting Augmentation

#### ColorJitter
- **Brightness**: `0.3` (reduced from `0.4`)
  - **Rationale**: Simulate exposure variations while preserving highlight texture details.
- **Contrast**: `0.3` (reduced from `0.4`)
  - **Rationale**: Simulate contrast reduction due to smoke/haze without losing texture contrast.
- **Saturation**: `0.2` (reduced from `0.4`)
  - **Rationale**: Excessive saturation changes can mask the inherent pale or fresh red characteristics of meat.
- **Hue**: `0.01` (significantly reduced from `0.1`)
  - **Rationale**: **Critical parameter**. For tasks relying on color discrimination, hue shifts can turn red meat into orange or purple, directly confusing category features. Almost locking hue preserves color-critical information.

### Environmental Simulation

#### RandomJPEG
- **Probability**: `0.2` (reduced from `0.4`)
- **Quality Range**: `(50, 95)` (improved from `(30, 80)`)
  - **Rationale**: Reduce compression distortion probability to preserve texture details. Higher minimum quality prevents excessive artifacts.

#### GaussianBlur
- **Kernel Size**: `(5, 9)` (reduced from `(7, 11)`)
- **Sigma Range**: `(0.1, 1.5)` (significantly reduced from `(0.1, 4.0)`)
- **Probability**: `0.3`
  - **Rationale**: Simulate slight haze/smoke effect. `sigma=4.0` is equivalent to severe myopia, which is destructive for texture recognition. The goal is "slight haze" rather than "complete defocus".

### Occlusion Handling

#### RandomErasing
- **Probability**: `0.3` (reduced from `0.4`)
- **Scale**: `(0.02, 0.1)` (reduced from `(0.02, 0.25)`)
- **Ratio**: `(0.3, 3.3)`
- **Value**: `"random"`
- **Removed**: Second RandomErasing (structured occlusion with `value=0`)
  - **Rationale**: Reduce occlusion area to prevent covering key texture regions. Single erasing is sufficient; double erasing may over-occlude critical features.

## Parameter Comparison

| Parameter | Previous Value | Current Value | Change Reason |
|-----------|---------------|---------------|---------------|
| RandomResizedCrop scale | (0.5, 1.2) | (0.8, 1.0) | Preserve context |
| RandomResizedCrop ratio | (0.75, 1.33) | (0.9, 1.1) | Maintain aspect ratio |
| RandomAffine degrees | 20 | 15 | Moderate rotation |
| RandomAffine translate | (0.15, 0.15) | (0.1, 0.1) | Moderate translation |
| RandomAffine scale | (0.8, 1.2) | (0.9, 1.1) | Converged range |
| RandomAffine shear | 10 | 0 | Remove non-physical distortion |
| RandomPerspective distortion | 0.3 | 0.1 | Slight viewpoint change |
| RandomPerspective p | 0.5 | 0.3 | Reduced probability |
| ColorJitter brightness | 0.4 | 0.3 | Preserve highlight details |
| ColorJitter contrast | 0.4 | 0.3 | Preserve texture contrast |
| ColorJitter saturation | 0.4 | 0.2 | Preserve color characteristics |
| ColorJitter hue | 0.1 | 0.01 | **Critical**: Lock hue for color-sensitive tasks |
| GaussianBlur sigma max | 4.0 | 1.5 | Slight haze, not defocus |
| RandomJPEG p | 0.4 | 0.2 | Reduce compression artifacts |
| RandomJPEG qmin | 30 | 50 | Higher minimum quality |
| RandomJPEG qmax | 80 | 95 | Higher maximum quality |
| RandomErasing p | 0.4 | 0.3 | Reduce occlusion |
| RandomErasing scale max | 0.25 | 0.1 | Protect key textures |
| RandomErasing count | 2 | 1 | Avoid over-occlusion |

## Usage

The augmentation pipeline is automatically applied during training when using `train_dinov2_arcface_small.py`. The `build_train_tfm()` function constructs the transformation pipeline based on the specified image size.

```python
train_transform = build_train_tfm(img_size=128)
```

## Future Improvements

If the current augmentation strategy still yields suboptimal results, consider:

1. **CutMix instead of RandomErasing**: For texture recognition, replacing erased black patches with patches from another image (CutMix) forces the model to focus on local texture details.

2. **Higher Resolution**: If hardware allows, use larger `img_size` (e.g., 448x448 instead of 224x224). Texture classification benefits significantly from higher resolution.

3. **Label Smoothing**: For visually similar categories, use label smoothing in the loss function to prevent overconfidence on incorrect samples.

4. **Adaptive Augmentation**: Consider class-specific augmentation strategies if different categories have different sensitivity to certain transformations.

## References

- Fine-Grained Visual Classification (FGVC) literature
- Data augmentation best practices for texture-sensitive tasks
- DINOv2 training guidelines
