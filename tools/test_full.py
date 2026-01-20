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
import re
import shutil
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
from eval import load_model, classify_batch, build_val_tfm
from test_dinov2_classification import load_label_file
import chardet
from utils.logger import logger_manager
from utils.file import mkdir_simple
import matplotlib

# Suppress matplotlib font warnings by setting font properties before importing pyplot
matplotlib.rcParams['font.family'] = 'DejaVu Sans'  # Use a common font to avoid warnings
matplotlib.rcParams['axes.unicode_minus'] = False  # Avoid unicode minus warnings

# Try to use interactive backend for displaying images
try:
    matplotlib.use('TkAgg')  # Use TkAgg backend for displaying images
except:
    try:
        matplotlib.use('Qt5Agg')
    except:
        pass  # Use default backend if both fail
import matplotlib.pyplot as plt

plt.ion()  # Turn on interactive mode
import numpy as np
import warnings

# Suppress matplotlib font-related warnings
warnings.filterwarnings('ignore', message='.*findfont.*')
warnings.filterwarnings('ignore', message='.*Generic family.*')

logger_manager.set_log_level(level="DEBUG")

# Security constants
MAX_FILE_SIZE = 1024 * 1024 * 1024  # 1GB
MAX_LABEL_FILE_SIZE = 10 * 1024 * 1024  # 10MB
MAX_CSV_FILE_SIZE = 500 * 1024 * 1024  # 500MB
MAX_IMAGE_FILE_SIZE = 100 * 1024 * 1024  # 100MB
ALLOWED_IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}
ALLOWED_LABEL_EXTENSIONS = {'.txt', '.csv'}
ALLOWED_MODEL_EXTENSIONS = {'.pt', '.pth', '.ckpt'}


def validate_file_path(file_path: str, check_exists: bool = True, check_readable: bool = True) -> tuple[bool, str]:
    """
    Validate file path for security.
    
    Args:
        file_path: Path to validate
        check_exists: Whether to check if file exists
        check_readable: Whether to check if file is readable
        
    Returns:
        (is_valid, error_message)
    """
    if not file_path or not isinstance(file_path, str):
        return False, "File path is empty or not a string"

    # Normalize path to prevent directory traversal
    try:
        normalized_path = os.path.normpath(os.path.abspath(file_path))
    except Exception as e:
        return False, f"Invalid file path: {e}"

    # Check if path exists
    if check_exists and not os.path.exists(normalized_path):
        return False, f"File does not exist: {normalized_path}"

    # Check if it's a file (not directory)
    if check_exists and not os.path.isfile(normalized_path):
        return False, f"Path is not a file: {normalized_path}"

    # Check if file is readable
    if check_readable and check_exists:
        if not os.access(normalized_path, os.R_OK):
            return False, f"File is not readable: {normalized_path}"

    return True, ""


def validate_file_size(file_path: str, max_size: int) -> tuple[bool, str]:
    """
    Validate file size.
    
    Args:
        file_path: Path to file
        max_size: Maximum allowed size in bytes
        
    Returns:
        (is_valid, error_message)
    """
    try:
        file_size = os.path.getsize(file_path)
        if file_size > max_size:
            size_mb = file_size / (1024 * 1024)
            max_mb = max_size / (1024 * 1024)
            return False, f"File size ({size_mb:.2f}MB) exceeds maximum allowed size ({max_mb:.2f}MB)"
        return True, ""
    except Exception as e:
        return False, f"Failed to check file size: {e}"


def validate_file_extension(file_path: str, allowed_extensions: set) -> tuple[bool, str]:
    """
    Validate file extension.
    
    Args:
        file_path: Path to file
        allowed_extensions: Set of allowed extensions (e.g., {'.txt', '.csv'})
        
    Returns:
        (is_valid, error_message)
    """
    ext = os.path.splitext(file_path)[1].lower()
    if ext not in allowed_extensions:
        return False, f"File extension '{ext}' is not allowed. Allowed extensions: {allowed_extensions}"
    return True, ""


def safe_read_file(file_path: str, file_type: str = "generic", max_size: int = None,
                   allowed_extensions: set = None, check_exists: bool = True) -> tuple[bool, str]:
    """
    Comprehensive file security check before reading.
    
    Args:
        file_path: Path to file
        file_type: Type of file for logging (e.g., "label", "csv", "model", "image")
        max_size: Maximum allowed file size in bytes
        allowed_extensions: Set of allowed file extensions
        check_exists: Whether to check if file exists
        
    Returns:
        (is_valid, error_message)
    """
    # Validate path
    is_valid, error_msg = validate_file_path(file_path, check_exists=check_exists, check_readable=True)
    if not is_valid:
        print(f"[Security] {file_type} file validation failed: {error_msg}", level="error")
        return False, error_msg

    normalized_path = os.path.normpath(os.path.abspath(file_path))

    # Check file extension if specified
    if allowed_extensions:
        is_valid, error_msg = validate_file_extension(normalized_path, allowed_extensions)
        if not is_valid:
            print(f"[Security] {file_type} file extension validation failed: {error_msg}", level="error")
            return False, error_msg

    # Check file size if specified
    if max_size:
        is_valid, error_msg = validate_file_size(normalized_path, max_size)
        if not is_valid:
            print(f"[Security] {file_type} file size validation failed: {error_msg}", level="error")
            return False, error_msg

    print(f"[Security] {file_type} file validation passed: {normalized_path}", level="debug")
    return True, ""


def parse_bbox(bbox_str: str, img_width: int, img_height: int):
    """
    Parse normalized bbox string to pixel coordinates.
    
    Args:
        bbox_str: Normalized bbox string "cx cy w h" (center point and width/height, values in [0, 1])
        img_width: Image width in pixels
        img_height: Image height in pixels
        
    Returns:
        (x, y, w, h) in pixel coordinates (top-left corner and width/height), or None if invalid
    """
    if pd.isna(bbox_str) or not bbox_str or str(bbox_str).strip() == '':
        print(f"[ParseBbox] Empty or NaN bbox string", level="debug")
        return None

    try:
        parts = str(bbox_str).strip().split()
        if len(parts) < 4:
            print(f"[ParseBbox] Invalid bbox format: '{bbox_str}' (expected 4 values, got {len(parts)})", level="debug")
            return None

        cx_norm = float(parts[0])  # Center x (normalized)
        cy_norm = float(parts[1])  # Center y (normalized)
        w_norm = float(parts[2])  # Width (normalized)
        h_norm = float(parts[3])  # Height (normalized)

        # Validate normalized coordinates
        if not (0 <= cx_norm <= 1 and 0 <= cy_norm <= 1 and 0 <= w_norm <= 1 and 0 <= h_norm <= 1):
            print(f"[ParseBbox] Bbox values out of range [0,1]: cx={cx_norm}, cy={cy_norm}, w={w_norm}, h={h_norm}",
                  level="debug")

        # Convert normalized coordinates to pixel coordinates
        cx = cx_norm * img_width  # Center x in pixels
        cy = cy_norm * img_height  # Center y in pixels
        w = w_norm * img_width  # Width in pixels
        h = h_norm * img_height  # Height in pixels

        # Convert center point to top-left corner
        x = cx - w / 2
        y = cy - h / 2

        # Ensure bbox is within image bounds
        x = max(0, min(int(x), img_width - 1))
        y = max(0, min(int(y), img_height - 1))
        w = max(1, min(int(w), img_width - x))
        h = max(1, min(int(h), img_height - y))

        return (x, y, w, h)
    except (ValueError, IndexError) as e:
        print(f"[ParseBbox] Failed to parse bbox '{bbox_str}': {e}", level="debug")
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
        print(f"[LoadImage] Image not found: {img_path}", level="debug")
        return None

    try:
        img = Image.open(img_path).convert('RGB')
    except Exception as e:
        print(f"[LoadImage] Failed to load image {img_path}: {e}", level="debug")
        return None

    return img


def crop_image_by_bbox(img, bbox_str: str):
    """
    Crop image using bbox coordinates.
    
    Args:
        img: PIL Image object
        bbox_str: Normalized bbox string "x y w h"
        
    Returns:
        PIL Image object of cropped region, or None if failed
    """
    img_width, img_height = img.size

    bbox = parse_bbox(bbox_str, img_width, img_height)
    if bbox is None:
        return None

    x, y, w, h = bbox
    # PIL crop expects (left, top, right, bottom)
    try:
        cropped = img.crop((x, y, x + w, y + h))
        return cropped
    except Exception as e:
        return None


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
        print(f"[SaveImage] Failed to save cropped image to {output_path}: {e}", level="debug")
        return False


def visualize_image_with_bbox(image_path: str, bbox_str: str, label: str = None, confidence: float = None,
                              base_dir: str = None, show_label: bool = True, display: bool = True):
    """
    Visualize original image with bbox, optionally with label and confidence.
    
    Args:
        image_path: Path to original image (can be relative or absolute)
        bbox_str: Normalized bbox string "x y w h"
        label: Predicted class label (optional, only used if show_label=True)
        confidence: Prediction confidence score (optional, only used if show_label=True)
        base_dir: Base directory for resolving relative paths
        show_label: Whether to show label and confidence text
        display: Whether to display the image using matplotlib
        
    Returns:
        PIL Image object with bbox drawn, or None if failed
    """
    try:
        # Resolve image path
        img_path = str(image_path).strip()
        if not os.path.isabs(img_path) and base_dir:
            img_path = os.path.join(base_dir, img_path)

        if not os.path.exists(img_path):
            return None

        # Load original image
        img = Image.open(img_path).convert('RGB')
        img_width, img_height = img.size

        # Parse bbox
        bbox = parse_bbox(bbox_str, img_width, img_height)
        if bbox is None:
            return None

        x, y, w, h = bbox

        # Create a copy for drawing
        img_draw = img.copy()
        draw = ImageDraw.Draw(img_draw)

        # Draw bbox rectangle
        bbox_color = (255, 0, 0)  # Red
        line_width = max(2, int(min(img_width, img_height) / 300))
        draw.rectangle([x, y, x + w, y + h], outline=bbox_color, width=line_width)

        # Draw label and confidence if requested
        if show_label and label is not None and confidence is not None:
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

        # Display image if requested
        if display:
            try:
                # Resize image if it exceeds 1080p (1920x1080) resolution
                max_width = 1920
                max_height = 1080
                display_img = img_draw

                if img_width > max_width or img_height > max_height:
                    # Calculate scaling factor to fit within 1080p while maintaining aspect ratio
                    scale_w = max_width / img_width
                    scale_h = max_height / img_height
                    scale = min(scale_w, scale_h)

                    new_width = int(img_width * scale)
                    new_height = int(img_height * scale)
                    # Use LANCZOS resampling (compatible with both old and new PIL versions)
                    try:
                        display_img = img_draw.resize((new_width, new_height), Image.Resampling.LANCZOS)
                    except AttributeError:
                        # Fallback for older PIL versions
                        display_img = img_draw.resize((new_width, new_height), Image.LANCZOS)
                    print(
                        f"[Visualize] Resized image from {img_width}x{img_height} to {new_width}x{new_height} for display",
                        level="debug")

                # Convert PIL Image to numpy array for matplotlib
                img_array = np.array(display_img)

                # Check if image is valid
                if img_array.size == 0:
                    print(f"[Warning] Empty image array for {img_path}", level="warning")
                    return img_draw

                fig = plt.figure(figsize=(12, 8))
                plt.imshow(img_array)
                plt.axis('off')
                plt.title(f"Image: {os.path.basename(img_path)}")
                plt.tight_layout()
                plt.show(block=False)
                plt.pause(0.1)  # Brief pause to allow display update
                plt.close(fig)  # Close specific figure to prevent blank windows
            except Exception as e:
                print(f"[Warning] Failed to display image: {e}", level="warning")

        return img_draw

    except Exception as e:
        print(f"[Warning] Failed to visualize image {image_path}: {e}", level="warning")
        return None


def generate_saved_filename(row, prefix: str, video_name_col: str = None, cross_frame_col: str = None) -> str:
    """
    Generate filename for saved image based on video name, frame number, and frame type.
    Format: video_name帧号_frame_type.jpg
    
    Args:
        row: pandas Series containing row data
        prefix: Field prefix (take_first, take_cross, return, return_static)
        video_name_col: Column name for video name (default: try common names)
        cross_frame_col: Column name for cross frame number (default: try common names)
        
    Returns:
        Generated filename string
    """
    # Map prefix to Chinese frame type names
    frame_type_map = {
        'take_first': '取走目标的第一帧',
        'take_cross': '取走目标的过线帧',
        'return': '放回目标的过线帧',
        'return_static': '放回目标的静止帧',
    }

    # Get frame type name
    frame_type = frame_type_map.get(prefix, prefix)

    # Try to get video name from various possible column names
    # Priority: video_name_col parameter > video_name > other common names
    video_name = "unknown_video"
    if video_name_col and video_name_col in row.index:
        video_name = str(row[video_name_col]).strip()
    else:
        # Try common column names, prioritize 'video_name' as per CSV schema
        for col_name in ['video_name', 'video_path', 'video', 'vid_name', 'vid_path']:
            if col_name in row.index and not pd.isna(row[col_name]):
                video_name = str(row[col_name]).strip()
                # Extract filename without extension if it's a path
                if os.path.sep in video_name:
                    video_name = os.path.splitext(os.path.basename(video_name))[0]
                break

    # Try to get frame number from various possible column names
    # Priority: cross_frame_col parameter > event_frame > other common names
    frame_number = "none"
    if cross_frame_col and cross_frame_col in row.index:
        frame_number = str(row[cross_frame_col]).strip()
    else:
        # Try common column names, prioritize 'event_frame' as per CSV schema
        for col_name in ['event_frame', 'cross_frame', 'frame_number', 'frame_num', 'cross_line_frame', 'frame']:
            if col_name in row.index and not pd.isna(row[col_name]):
                frame_number = str(int(row[col_name])) if pd.api.types.is_number(row[col_name]) else str(
                    row[col_name]).strip()
                break

    # Generate filename: video_name帧号_frame_type.jpg
    filename = f"{video_name}_{frame_number}_{frame_type}.jpg"

    # Sanitize filename (remove invalid characters)
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        filename = filename.replace(char, '_')

    return filename


def collect_all_images(df, base_dir: str, temp_dir: str, visualize: bool = False):
    """
    Collect all images that need to be cropped and classified from the dataframe.
    
    Args:
        df: pandas DataFrame
        base_dir: Base directory for resolving image paths
        temp_dir: Temporary directory for saving cropped images
        visualize: Whether to visualize images (unused, kept for compatibility)
        
    Returns:
        tasks: List of tuples (row_idx, field_prefix, temp_path, orig_img_path, bbox_str, row_data) for successful crops
        failed_tasks: List of tuples (row_idx, field_prefix, reason) for failed crops
    """
    tasks = []
    failed_tasks = []

    # Statistics for failure reasons
    failure_stats = {
        'image_not_found': 0,
        'image_load_error': 0,
        'bbox_parse_error': 0,
        'crop_error': 0,
        'save_error': 0,
        'missing_column': 0,
        'empty_image_name': 0,
    }

    # Define the 4 image fields to process
    image_fields = [
        ('take_first_image_name', 'take_first_bbox', 'take_first'),
        ('take_cross_image_name', 'take_cross_bbox', 'take_cross'),
        ('return_image_name', 'return_bbox', 'return'),
        ('return_static_image_name', 'return_static_bbox', 'return_static'),
    ]

    print(f"Collecting and cropping images from {len(df)} rows...", level="info")
    print(f"Base directory: {base_dir}", level="info")
    print(f"Temp directory: {temp_dir}", level="info")

    for row_idx, row in tqdm(df.iterrows(), total=len(df), desc="Cropping images"):
        for img_name_col, bbox_col, prefix in image_fields:
            # Check if columns exist
            if img_name_col not in row.index or bbox_col not in row.index:
                failure_stats['missing_column'] += 1
                failed_tasks.append((row_idx, prefix, 'missing_column'))
                print(f"[Row {row_idx}, {prefix}] Missing columns: {img_name_col} or {bbox_col}", level="debug")
                continue

            img_name = row[img_name_col]
            bbox_str = row[bbox_col]

            # Check if image name is empty
            if pd.isna(img_name) or not str(img_name).strip():
                failure_stats['empty_image_name'] += 1
                failed_tasks.append((row_idx, prefix, 'empty_image_name'))
                continue

            # Resolve original image path for visualization
            orig_img_path = str(img_name).strip()
            if not os.path.isabs(orig_img_path) and base_dir:
                orig_img_path = os.path.join(base_dir, orig_img_path)

            # Load image
            img = load_image(img_name, base_dir)
            if img is None:
                failure_stats['image_not_found'] += 1
                failed_tasks.append((row_idx, prefix, 'image_not_found'))
                print(f"[Row {row_idx}, {prefix}] Failed to load image: {orig_img_path}", level="debug")
                continue

            # Crop image
            cropped_img = crop_image_by_bbox(img, bbox_str)
            if cropped_img is None:
                failure_stats['bbox_parse_error'] += 1
                failed_tasks.append((row_idx, prefix, 'bbox_parse_error'))
                print(f"[Row {row_idx}, {prefix}] Failed to crop image: bbox='{bbox_str}', image={orig_img_path}",
                      level="debug")
                continue

            # Save cropped image temporarily
            temp_filename = f"row_{row_idx}_{prefix}_{os.path.basename(str(img_name))}"
            temp_path = os.path.join(temp_dir, temp_filename)
            if save_cropped_image(cropped_img, temp_path):
                # Store row data for later use in filename generation
                tasks.append((row_idx, prefix, temp_path, orig_img_path, bbox_str, row))

                # Visualize during loading phase (only bbox, no label/confidence)
                # if visualize:
                #     visualize_image_with_bbox(orig_img_path, bbox_str, label=None, confidence=None, base_dir=base_dir,
                #                               show_label=False, display=True)
            else:
                failure_stats['save_error'] += 1
                failed_tasks.append((row_idx, prefix, 'save_error'))
                print(f"[Row {row_idx}, {prefix}] Failed to save cropped image: {temp_path}", level="debug")

    # Print detailed statistics
    print(f"Successfully cropped {len(tasks)} images, {len(failed_tasks)} failed", level="info")
    print("Failure statistics:", level="info")
    for reason, count in failure_stats.items():
        if count > 0:
            print(f"  {reason}: {count}", level="info")

    # Print sample failures for debugging
    if len(failed_tasks) > 0:
        print("Sample failures (first 10):", level="info")
        for i, (row_idx, prefix, reason) in enumerate(failed_tasks[:10]):
            img_name_col = f"{prefix}_image_name"
            bbox_col = f"{prefix}_bbox"
            if img_name_col in df.columns and bbox_col in df.columns:
                img_name = df.at[row_idx, img_name_col]
                bbox_str = df.at[row_idx, bbox_col]
                print(f"  Row {row_idx}, {prefix}: reason={reason}, image='{img_name}', bbox='{bbox_str}'",
                      level="info")

    return tasks, failed_tasks


def read_csv_auto(
        path,
        encodings=("utf-8", "utf-8-sig", "gb18030", "gbk", "latin1"),
        sample_size=100_000,
        **kwargs
):
    """
    自动识别 CSV 编码并读取
    - 先用 chardet 探测
    - 再按候选编码顺序 fallback
    - 永不重复传 encoding
    """

    kwargs.pop("encoding", None)

    try:
        if not os.path.isfile(path):
            print("CSV file does not exist: {}".format(path), level="error")
            exit(0)

        print(f"[Info] Reading CSV from {path}")
        with open(path, "rb") as f:
            raw = f.read(sample_size)
        detected = chardet.detect(raw)
        if detected["encoding"]:
            encodings = (detected["encoding"],) + tuple(
                e for e in encodings if e != detected["encoding"]
            )
    except Exception as e:
        print(f"Encoding detection failed: {e}", level="warning")

    last_err = None
    for enc in encodings:
        try:
            print(f"Trying CSV encoding: {enc}", level="debug")
            return pd.read_csv(path, encoding=enc, **kwargs)
        except Exception as e:
            last_err = e

    raise RuntimeError(f"Failed to read CSV {path}") from last_err


def main():
    parser = argparse.ArgumentParser(
        description="Batch inference tool: read CSV with image paths and bboxes, classify cropped images, output results",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--input_csv", type=str, required=True,
                        help="Input CSV file path")
    parser.add_argument("--suffix", type=str, default='label',
                        help="Output CSV file suffix name, default is label")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to model checkpoint (.pt file)")
    parser.add_argument("--label_file", type=str, default=None,
                        help="Path to label file. Format: label_id class_name (one per line). "
                             "If not provided, will use Class_{id} as fallback")
    parser.add_argument("--base_dir", type=str, default="",
                        help="Base directory for resolving relative image paths, default is the directory of input_csv")
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
    parser.add_argument("--temp_save_dir", type=str,
                        help="Directory to save cropped images organized by category")
    args = parser.parse_args()

    # Setup device
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
        print("[Warning] CUDA not available, using CPU")

    version = os.path.basename(os.path.dirname(args.model_path))
    category = os.path.basename(os.path.dirname(args.input_csv))
    print(f"-> Version: {version}, Category: {category}", level="info")

    # Load class names mapping
    classes = load_label_file(args.label_file)

    # Load model
    print(f"[Info] Loading model from {args.model_path}")
    embedder, head, model_args = load_model(args.model_path, device)

    # Build transform
    img_size = model_args.get("img_size", 128)
    transform = build_val_tfm(img_size)

    # Read input CSV
    try:
        df = read_csv_auto(args.input_csv)
    except Exception as e:
        print(f"[Error] Failed to read CSV: {e}")
        return

    print(f"[Info] Loaded {len(df)} rows from CSV")

    # Create temp directory
    os.makedirs(args.temp_dir, exist_ok=True)

    # Step 1: Collect and crop all images
    base_dir = args.base_dir if args.base_dir else os.path.dirname(args.input_csv)
    tasks, failed_tasks = collect_all_images(df, base_dir, args.temp_dir, args.visualize)

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

    output_csv = os.path.splitext(args.input_csv)[0] + f"_{args.suffix}.csv"
    mkdir_simple(output_csv)

    # Map successful results (using class names instead of IDs)
    # Also save images by category if requested
    if args.temp_save_dir:
        temp_save_dir = os.path.join(args.temp_save_dir, category, version)
        os.makedirs(temp_save_dir, exist_ok=True)
        print(f"[Info] Saving cropped images by category to {args.temp_save_dir}...")

    for task, (pred_class, confidence) in zip(tasks, all_results):
        row_idx, prefix, temp_path, orig_img_path, bbox_str, row_data = task
        if pred_class is not None:
            # Get class name from mapping, fallback to Class_{id} if not found
            class_name = classes[pred_class]
            df.at[row_idx, f'{prefix}_image_label'] = class_name
            df.at[row_idx, f'{prefix}_image_confidence'] = float(confidence)

            # Visualize during evaluation phase (with bbox, label and confidence)
            if args.visualize:
                visualize_image_with_bbox(
                    orig_img_path,
                    bbox_str,
                    label=class_name,
                    confidence=float(confidence),
                    base_dir=base_dir,
                    show_label=True,
                    display=True
                )

            # Save image by category if requested
            if args.temp_save_dir:
                try:
                    # Create category directory
                    category_dir = os.path.join(temp_save_dir, class_name)
                    os.makedirs(category_dir, exist_ok=True)

                    # Generate filename with video name, cross frame, and frame attributes
                    saved_filename = generate_saved_filename(row_data, prefix)
                    saved_path = os.path.join(category_dir, saved_filename)

                    # Copy the cropped image to the category directory
                    if os.path.exists(temp_path):
                        shutil.copy2(temp_path, saved_path)
                except Exception as e:
                    print(f"[Warning] Failed to save image by category: {e}", level="warning")

    # Save output CSV
    print(f"[Info] Saving results to {output_csv}")
    try:
        df.to_csv(output_csv, index=False, encoding='utf-8-sig')
        print(f"[Info] Successfully saved {len(df)} rows to {output_csv}")
    except Exception as e:
        print(f"[Error] Failed to save CSV: {e}")
        return

    # Cleanup temp directory (optional)
    shutil.rmtree(args.temp_dir)
    print(f"[Info] Cleaned up temporary directory {args.temp_dir}")

    print("[Info] Processing completed!")


if __name__ == '__main__':
    main()
