"""Layout region detection module for design-to-HTML converter.

This module analyzes image brightness to identify and locate rectangular layout regions:
- Header: top 15% of image
- Footer: bottom 15% of image
- Sidebar: left 25% of image
- Content: remaining central region

Detection uses brightness analysis: a region is detected if its average brightness
differs by more than 15% from the global image mean brightness.
"""

import numpy as np
import sys
from pathlib import Path

# Load types.py explicitly by file path to avoid conflict with stdlib types
project_root = Path(__file__).parent
types_path = project_root / "types.py"
import importlib.util
spec = importlib.util.spec_from_file_location("project_types", str(types_path))
project_types = importlib.util.module_from_spec(spec)
sys.modules['project_types'] = project_types
spec.loader.exec_module(project_types)

LayoutDetectionError = project_types.LayoutDetectionError


def detect_layout_regions(image: np.ndarray) -> dict:
    """
    Detect header, sidebar, content, and footer regions via brightness analysis.

    Args:
        image: NumPy array of shape (H, W, 3), dtype uint8, RGB 0-255

    Returns:
        Dict with keys: 'header', 'sidebar', 'content', 'footer'
        Each value is either None or {'x': int, 'y': int, 'width': int, 'height': int}

        Example:
        {
            'header': {'x': 0, 'y': 0, 'width': 800, 'height': 120},
            'sidebar': {'x': 0, 'y': 120, 'width': 200, 'height': 480},
            'content': {'x': 200, 'y': 120, 'width': 600, 'height': 480},
            'footer': {'x': 0, 'y': 600, 'width': 800, 'height': 100}
        }

    Raises:
        LayoutDetectionError: If detection fails (rare; mostly for validation)

    Algorithm:
    1. Convert image to grayscale using luminance formula
    2. Compute global mean brightness and 15% threshold
    3. Check header, footer, sidebar regions for brightness difference
    4. Calculate content region from remaining space
    5. Validate all coordinates and return results
    """
    try:
        # Get image dimensions
        image_height, image_width = image.shape[0], image.shape[1]

        # Step 1: Convert to grayscale using standard luminance formula
        # gray = 0.299*R + 0.587*G + 0.114*B
        gray = (
            0.299 * image[:, :, 0].astype(np.float32)
            + 0.587 * image[:, :, 1].astype(np.float32)
            + 0.114 * image[:, :, 2].astype(np.float32)
        )

        # Step 2: Compute global mean brightness and threshold
        global_mean = np.mean(gray)
        threshold = global_mean * 0.15  # 15% of global mean

        # Step 3: Header detection (top 15%)
        header_height = int(np.ceil(0.15 * image_height))
        header = _detect_horizontal_region(
            gray, 0, 0, header_height, image_width, global_mean, threshold
        )
        if header is not None:
            header["x"] = 0
            header["y"] = 0
            header["width"] = image_width

        # Step 4: Footer detection (bottom 15%)
        footer_y_start = int(np.floor(0.85 * image_height))
        footer_height = image_height - footer_y_start
        footer = _detect_horizontal_region(
            gray, footer_y_start, 0, footer_height, image_width, global_mean, threshold
        )
        if footer is not None:
            footer["x"] = 0
            footer["y"] = footer_y_start
            footer["width"] = image_width

        # Step 5: Sidebar detection (left 25%)
        sidebar_width = int(np.ceil(0.25 * image_width))
        sidebar = _detect_vertical_region(
            gray, 0, 0, image_height, sidebar_width, global_mean, threshold
        )
        if sidebar is not None:
            sidebar["x"] = 0
            sidebar["y"] = 0
            sidebar["height"] = image_height

        # Step 6: Content detection (central region after header/sidebar/footer)
        content_x = sidebar_width if sidebar is not None else 0
        content_y = header_height if header is not None else 0
        content_y_end = footer_y_start if footer is not None else image_height
        content_width = image_width - content_x
        content_height = content_y_end - content_y

        content = None
        if content_width > 0 and content_height > 0:
            content = {
                "x": content_x,
                "y": content_y,
                "width": content_width,
                "height": content_height,
            }

        # Step 7: Validate all coordinates
        result = {
            "header": header,
            "sidebar": sidebar,
            "content": content,
            "footer": footer,
        }

        # Validate bounds for all detected regions
        for region_name, region in result.items():
            if region is not None:
                _validate_region_bounds(region, image_width, image_height)

        return result

    except LayoutDetectionError:
        # Re-raise validation errors
        raise
    except Exception as e:
        # Wrap unexpected errors
        raise LayoutDetectionError(
            f"Layout detection failed: {str(e)}"
        ) from e


def _detect_horizontal_region(
    gray: np.ndarray,
    y_start: int,
    x_start: int,
    height: int,
    width: int,
    global_mean: float,
    threshold: float,
) -> dict | None:
    """
    Detect a horizontal region (header/footer) via brightness analysis.

    Samples center 80% of the region (ignores 10% edge margins).

    Args:
        gray: Grayscale image array
        y_start: Top edge of region
        x_start: Left edge of region
        height: Height of region
        width: Width of region
        global_mean: Global mean brightness
        threshold: Detection threshold (15% of global mean)

    Returns:
        {'height': int} if detected, None otherwise
    """
    # Sample center 80% (ignore 10% margins on all sides)
    y_margin = int(np.ceil(0.1 * height))
    x_margin = int(np.ceil(0.1 * width))

    y_sample_start = y_start + y_margin
    y_sample_end = y_start + height - int(np.floor(0.1 * height))
    x_sample_start = x_start + x_margin
    x_sample_end = x_start + width - int(np.floor(0.1 * width))

    # Ensure valid sampling bounds
    y_sample_start = max(y_sample_start, y_start)
    y_sample_end = min(y_sample_end, y_start + height)
    x_sample_start = max(x_sample_start, x_start)
    x_sample_end = min(x_sample_end, x_start + width)

    if y_sample_start >= y_sample_end or x_sample_start >= x_sample_end:
        return None

    # Extract sampled pixels
    sampled = gray[y_sample_start:y_sample_end, x_sample_start:x_sample_end]
    region_brightness = np.mean(sampled)

    # Check if brightness differs by more than threshold
    if abs(region_brightness - global_mean) > threshold:
        return {"height": height}

    return None


def _detect_vertical_region(
    gray: np.ndarray,
    y_start: int,
    x_start: int,
    height: int,
    width: int,
    global_mean: float,
    threshold: float,
) -> dict | None:
    """
    Detect a vertical region (sidebar) via brightness analysis.

    Samples center 80% of the region (ignores 10% edge margins).

    Args:
        gray: Grayscale image array
        y_start: Top edge of region
        x_start: Left edge of region
        height: Height of region
        width: Width of region
        global_mean: Global mean brightness
        threshold: Detection threshold (15% of global mean)

    Returns:
        {'width': int} if detected, None otherwise
    """
    # Sample center 80% (ignore 10% margins on all sides)
    y_margin = int(np.ceil(0.1 * height))
    x_margin = int(np.ceil(0.1 * width))

    y_sample_start = y_start + y_margin
    y_sample_end = y_start + height - int(np.floor(0.1 * height))
    x_sample_start = x_start + x_margin
    x_sample_end = x_start + width - int(np.floor(0.1 * width))

    # Ensure valid sampling bounds
    y_sample_start = max(y_sample_start, y_start)
    y_sample_end = min(y_sample_end, y_start + height)
    x_sample_start = max(x_sample_start, x_start)
    x_sample_end = min(x_sample_end, x_start + width)

    if y_sample_start >= y_sample_end or x_sample_start >= x_sample_end:
        return None

    # Extract sampled pixels
    sampled = gray[y_sample_start:y_sample_end, x_sample_start:x_sample_end]
    region_brightness = np.mean(sampled)

    # Check if brightness differs by more than threshold
    if abs(region_brightness - global_mean) > threshold:
        return {"width": width}

    return None


def _validate_region_bounds(
    region: dict, image_width: int, image_height: int
) -> None:
    """
    Validate region coordinates satisfy all constraints.

    Raises:
        LayoutDetectionError: If any constraint is violated
    """
    x = region.get("x", 0)
    y = region.get("y", 0)
    width = region.get("width", 0)
    height = region.get("height", 0)

    # Check all values are non-negative
    if x < 0 or y < 0 or width <= 0 or height <= 0:
        raise LayoutDetectionError(
            f"Invalid region bounds: x={x}, y={y}, width={width}, height={height}. "
            f"Require x >= 0, y >= 0, width > 0, height > 0"
        )

    # Check all values fit within image
    if x > image_width:
        raise LayoutDetectionError(
            f"Region x={x} exceeds image_width={image_width}"
        )
    if y > image_height:
        raise LayoutDetectionError(
            f"Region y={y} exceeds image_height={image_height}"
        )
    if x + width > image_width:
        raise LayoutDetectionError(
            f"Region x + width = {x + width} exceeds image_width={image_width}"
        )
    if y + height > image_height:
        raise LayoutDetectionError(
            f"Region y + height = {y + height} exceeds image_height={image_height}"
        )
