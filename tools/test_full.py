#!/usr/bin/env python3
# encoding: utf-8
'''
@author: 孙昊
@contact: smartadpole@163.com
@file: test_full.py
@time: 2026/1/16 16:29
@desc: Batch inference tool - read CSV with image paths and bboxes, crop images, classify, and output results to CSV
'''
import sys
import os

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

import argparse
import pandas as pd
import torch
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
from eval import load_model, classify_batch, build_val_tfm
from test_dinov2_classification import load_label_file
from utils.logger import logger_manager, Logger
logger_manager.set_log_level(level="DEBUG")

def parse_bbox(bbox_str: str, img_width: int, img_height: int):
    """
    Parse normalized bbox string to pixel coordinates.
    
    Args:
        bbox_str: Normalized bbox string "x y w h" (values in [0, 1])
        img_width: Image width in pixels
        img_height: Image height in pixels
        
    Returns:
        (x, y, w, h) in pixel coordinates, or None if invalid
    """
    if pd.isna(bbox_str) or not bbox_str or str(bbox_str).strip() == '':
        return None
    
    try:
        parts = str(bbox_str).strip().split()
        if len(parts) < 4:
            return None
        
        x_norm = float(parts[0])
        y_norm = float(parts[1])
        w_norm = float(parts[2])
        h_norm = float(parts[3])
        
        # Convert normalized coordinates to pixel coordinates
        x = int(x_norm * img_width)
        y = int(y_norm * img_height)
        w = int(w_norm * img_width)
        h = int(h_norm * img_height)
        
        # Ensure bbox is within image bounds
        x = max(0, min(x, img_width - 1))
        y = max(0, min(y, img_height - 1))
        w = max(1, min(w, img_width - x))
        h = max(1, min(h, img_height - y))
        
        return (x, y, w, h)
    except (ValueError, IndexError) as e:
        print(f"[Warning] Failed to parse bbox '{bbox_str}': {e}")
        return None

def load_image(image_path: str, base_dir: str = None):
    """
    Load image from path.

    Args:
        image_path: Path to image file (can be relative or absolute)
        base_dir: Base directory for resolving relative paths
    """

    img_path = str(image_path).strip()
    if not os.path.isabs(img_path) and base_dir:
        img_path = os.path.join(base_dir, img_path)

    if not os.path.exists(img_path):
        return None

    try:
        img = Image.open(img_path).convert('RGB')
    except Exception as e:
        print(f"[Warning] Failed to load image {img_path}: {e}")
        return None

    return img


def crop_image_by_bbox(img, bbox_str: str):
    """
    Crop image using bbox coordinates.
    
    Args:
        image_path: Path to image file (can be relative or absolute)
        bbox_str: Normalized bbox string "x y w h"
        base_dir: Base directory for resolving relative paths
        
    Returns:
        PIL Image object of cropped region, or None if failed
    """
    img_width, img_height = img.size

    bbox = parse_bbox(bbox_str, img_width, img_height)
    if bbox is None:
        return None

    x, y, w, h = bbox
    # PIL crop expects (left, top, right, bottom)
    cropped = img.crop((x, y, x + w, y + h))
    return cropped


def save_cropped_image(cropped_img: Image.Image, output_path: str):
    """
    Save cropped image to temporary file.
    
    Args:
        cropped_img: PIL Image object
        output_path: Path to save the cropped image
        
    Returns:
        True if successful, False otherwise
    """
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cropped_img.save(output_path)
        return True
    except Exception as e:
        print(f"[Warning] Failed to save cropped image to {output_path}: {e}")
        return False


def visualize_image_with_bbox(image_path: str, bbox_str: str, label: str, confidence: float, base_dir: str = None):
    """
    Visualize original image with bbox, label and confidence.
    
    Args:
        image_path: Path to original image (can be relative or absolute)
        bbox_str: Normalized bbox string "x y w h"
        label: Predicted class label
        confidence: Prediction confidence score
        base_dir: Base directory for resolving relative paths
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Resolve image path
        img_path = str(image_path).strip()
        if not os.path.isabs(img_path) and base_dir:
            img_path = os.path.join(base_dir, img_path)
        
        if not os.path.exists(img_path):
            return False
        
        # Load original image
        img = Image.open(img_path).convert('RGB')
        img_width, img_height = img.size
        
        # Parse bbox
        bbox = parse_bbox(bbox_str, img_width, img_height)
        if bbox is None:
            return False
        
        x, y, w, h = bbox
        
        # Create a copy for drawing
        img_draw = img.copy()
        draw = ImageDraw.Draw(img_draw)
        
        # Draw bbox rectangle
        bbox_color = (255, 0, 0)  # Red
        line_width = max(2, int(min(img_width, img_height) / 300))
        draw.rectangle([x, y, x + w, y + h], outline=bbox_color, width=line_width)
        
        # Prepare label text
        label_text = f"{label}: {confidence:.3f}"
        
        # Try to load a font, fallback to default if not available
        font = None
        font_size = max(16, int(min(img_width, img_height) / 40))
        
        # Try different font paths for different platforms
        font_paths = []
        
        # Windows font paths
        if os.name == 'nt':
            windir = os.environ.get('WINDIR', 'C:\\Windows')
            font_paths.extend([
                os.path.join(windir, 'Fonts', 'arial.ttf'),
                os.path.join(windir, 'Fonts', 'Arial.ttf'),
                os.path.join(windir, 'Fonts', 'msyh.ttc'),  # Microsoft YaHei (Chinese support)
            ])
        # Linux font paths
        elif os.name == 'posix':
            font_paths.extend([
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
            ])
        # macOS font paths
        if sys.platform == 'darwin':
            font_paths.extend([
                "/System/Library/Fonts/Helvetica.ttc",
                "/Library/Fonts/Arial.ttf",
            ])
        
        for font_path in font_paths:
            try:
                if os.path.exists(font_path):
                    font = ImageFont.truetype(font_path, font_size)
                    break
            except:
                continue
        
        # Fallback to default font if no truetype font found
        if font is None:
            try:
                font = ImageFont.load_default()
            except:
                font = None
        
        # Calculate text size (compatible with different PIL versions)
        if font:
            try:
                # PIL 9.0.0+ uses textbbox
                bbox_text = draw.textbbox((0, 0), label_text, font=font)
                text_width = bbox_text[2] - bbox_text[0]
                text_height = bbox_text[3] - bbox_text[1]
            except AttributeError:
                # Older PIL versions use textsize
                try:
                    text_width, text_height = draw.textsize(label_text, font=font)
                except:
                    # Fallback approximation
                    text_width = len(label_text) * 8
                    text_height = 16
        else:
            # Approximate text size for default font
            text_width = len(label_text) * 6
            text_height = 12
        
        # Draw text background
        text_x = x
        text_y = y - text_height - 4
        if text_y < 0:
            text_y = y + h + 4
        
        # Ensure text is within image bounds
        if text_x + text_width > img_width:
            text_x = img_width - text_width - 2
        if text_y + text_height > img_height:
            text_y = y - text_height - 4
            if text_y < 0:
                text_y = 2
        
        # Draw text background rectangle
        # Note: PIL ImageDraw doesn't support alpha channel directly, so we use solid color
        bg_color = (0, 0, 0)  # Black background
        draw.rectangle([text_x - 2, text_y - 2, text_x + text_width + 2, text_y + text_height + 2], 
                      fill=bg_color)
        
        # Draw text
        text_color = (255, 255, 255)  # White
        draw.text((text_x, text_y), label_text, fill=text_color, font=font)

        return True
        
    except Exception as e:
        print(f"[Warning] Failed to visualize image {image_path}: {e}")
        return False


def collect_all_images(df, base_dir: str, temp_dir: str, visualize: bool = False):
    """
    Collect all images that need to be cropped and classified from the dataframe.
    
    Args:
        df: pandas DataFrame
        base_dir: Base directory for resolving image paths
        temp_dir: Temporary directory for saving cropped images
        
    Returns:
        tasks: List of tuples (row_idx, field_prefix, temp_path, orig_img_path, bbox_str) for successful crops
        failed_tasks: List of tuples (row_idx, field_prefix) for failed crops
    """
    tasks = []
    failed_tasks = []
    
    # Define the 4 image fields to process
    image_fields = [
        ('take_first_image_name', 'take_first_bbox', 'take_first'),
        ('take_cross_image_name', 'take_cross_bbox', 'take_cross'),
        ('return_image_name', 'return_bbox', 'return'),
        ('return_static_image_name', 'return_static_bbox', 'return_static'),
    ]
    
    print(f"[Info] Collecting and cropping images from {len(df)} rows...")
    for row_idx, row in tqdm(df.iterrows(), total=len(df), desc="Cropping images"):
        for img_name_col, bbox_col, prefix in image_fields:
            if img_name_col not in row.index or bbox_col not in row.index:
                continue
            
            img_name = row[img_name_col]
            bbox_str = row[bbox_col]
            
            # Resolve original image path for visualization
            orig_img_path = str(img_name).strip()
            if not os.path.isabs(orig_img_path) and base_dir:
                orig_img_path = os.path.join(base_dir, orig_img_path)
            
            # Crop image
            img = load_image(img_name, base_dir)
            if img is None:
                failed_tasks.append((row_idx, prefix))
                continue
            cropped_img = crop_image_by_bbox(img, bbox_str)
            if cropped_img is None:
                failed_tasks.append((row_idx, prefix))
                continue
            
            # Save cropped image temporarily
            temp_filename = f"row_{row_idx}_{prefix}_{os.path.basename(str(img_name))}"
            temp_path = os.path.join(temp_dir, temp_filename)
            if save_cropped_image(cropped_img, temp_path):
                tasks.append((row_idx, prefix, temp_path, orig_img_path, bbox_str))
            else:
                failed_tasks.append((row_idx, prefix))
    
    print(f"[Info] Successfully cropped {len(tasks)} images, {len(failed_tasks)} failed")
    return tasks, failed_tasks


def main():
    parser = argparse.ArgumentParser(
        description="Batch inference tool: read CSV with image paths and bboxes, classify cropped images, output results",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--input_csv", type=str, required=True,
                        help="Input CSV file path")
    parser.add_argument("--output_csv", type=str, required=True,
                        help="Output CSV file path")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to model checkpoint (.pt file)")
    parser.add_argument("--label_file", type=str, default=None,
                        help="Path to label file. Format: label_id class_name (one per line). "
                             "If not provided, will use Class_{id} as fallback")
    parser.add_argument("--base_dir", type=str, default="",
                        help="Base directory for resolving relative image paths")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"],
                        help="Device to run inference on")
    parser.add_argument("--batch_size", "-b", type=int, default=32,
                        help="Batch size for inference")
    parser.add_argument("--temp_dir", type=str, default="temp_cropped",
                        help="Temporary directory for saving cropped images")
    parser.add_argument("--visualize", action='store_true', default=False,
                        help="Whether to visualize images with bbox, label and confidence")
    parser.add_argument("--vis_output_dir", type=str, default="visualizations",
                        help="Output directory for visualized images (only used if --visualize is set)")
    args = parser.parse_args()
    
    # Setup device
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
        print("[Warning] CUDA not available, using CPU")
    
    # Load class names mapping
    classes = load_label_file(args.label_file)
    
    # Load model
    print(f"[Info] Loading model from {args.model_path}")
    embedder, head, model_args = load_model(args.model_path, device)
    
    # Build transform
    img_size = model_args.get("img_size", 128)
    transform = build_val_tfm(img_size)
    
    # Read input CSV
    print(f"[Info] Reading CSV from {args.input_csv}")
    try:
        df = pd.read_csv(args.input_csv, encoding='utf-8')
    except Exception as e:
        print(f"[Error] Failed to read CSV: {e}")
        return
    
    print(f"[Info] Loaded {len(df)} rows from CSV")
    
    # Create temp directory
    os.makedirs(args.temp_dir, exist_ok=True)
    
    # Step 1: Collect and crop all images
    tasks, failed_tasks = collect_all_images(df, args.base_dir, args.temp_dir, args.visualize)
    
    if len(tasks) == 0:
        print("[Warning] No images were successfully cropped. Check your CSV and image paths.")
        return
    
    # Step 2: Batch classify all cropped images
    print(f"[Info] Classifying {len(tasks)} cropped images...")
    all_image_paths = [task[2] for task in tasks]
    all_results = []
    
    # Process in batches
    num_batches = (len(all_image_paths) + args.batch_size - 1) // args.batch_size
    with tqdm(total=len(all_image_paths), desc="Classifying") as pbar:
        for batch_idx in range(num_batches):
            start_idx = batch_idx * args.batch_size
            end_idx = min(start_idx + args.batch_size, len(all_image_paths))
            batch_paths = all_image_paths[start_idx:end_idx]
            
            batch_results = classify_batch(embedder, head, batch_paths, transform, device)
            all_results.extend(batch_results)
            
            pbar.update(len(batch_paths))

            # if visualize:
            #     visualize_image_with_bbox(orig_img_path, bbox_str, class_name,
            #                               float(confidence), args.base_dir)
    
    # Step 3: Map results back to dataframe
    print(f"[Info] Mapping results back to dataframe...")
    
    # Initialize result columns
    result_columns = [
        'take_first_image_label', 'take_first_image_confidence',
        'take_cross_image_label', 'take_cross_image_confidence',
        'return_image_label', 'return_image_confidence',
        'return_static_image_label', 'return_static_image_confidence',
    ]
    for col in result_columns:
        df[col] = None
    
    # Map successful results (using class names instead of IDs)
    for task, (pred_class, confidence) in zip(tasks, all_results):
        row_idx, prefix, _, orig_img_path, bbox_str = task
        if pred_class is not None:
            # Get class name from mapping, fallback to Class_{id} if not found
            class_name = classes[pred_class]
            df.at[row_idx, f'{prefix}_image_label'] = class_name
            df.at[row_idx, f'{prefix}_image_confidence'] = float(confidence)

    # Save output CSV
    print(f"[Info] Saving results to {args.output_csv}")
    try:
        df.to_csv(args.output_csv, index=False, encoding='utf-8-sig')
        print(f"[Info] Successfully saved {len(df)} rows to {args.output_csv}")
    except Exception as e:
        print(f"[Error] Failed to save CSV: {e}")
        return
    
    # Cleanup temp directory (optional)
    # import shutil
    # shutil.rmtree(args.temp_dir)
    # print(f"[Info] Cleaned up temporary directory {args.temp_dir}")
    
    print("[Info] Processing completed!")


if __name__ == '__main__':
    main()

