"""Drop-in replacement for layout_detector.detect_layout_regions() using ML model.

This module provides ml_detect_layout_regions() which has identical signature and
return type as the brightness-based detector, but uses a trained neural network.
"""

import numpy as np
import sys
from pathlib import Path
import importlib.util

# Load types module for LayoutDetectionError
if 'project_types' not in sys.modules:
    types_path = Path(__file__).parent.parent / "sources/38c98b3d/types.py"
    spec = importlib.util.spec_from_file_location("project_types", str(types_path))
    project_types = importlib.util.module_from_spec(spec)
    sys.modules['project_types'] = project_types
    spec.loader.exec_module(project_types)
else:
    project_types = sys.modules['project_types']

LayoutDetectionError = project_types.LayoutDetectionError

# Import inference engine
from .inference import ComponentClassifierInference

# Module-level cache for loaded model (singleton pattern)
_inference_engine: ComponentClassifierInference | None = None
_model_path: str | None = None


def ml_detect_layout_regions(
    image: np.ndarray,
    model_path: str = "./model.pt"
) -> dict:
    """Detect header, sidebar, content, and footer regions via learned neural network.

    This function has identical signature and return type as layout_detector.detect_layout_regions(),
    making it a drop-in replacement for the brightness-based detector.

    Args:
        image: NumPy array of shape (H, W, 3), dtype uint8, RGB 0-255
        model_path: Path to trained model checkpoint (default "./model.pt")

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
        LayoutDetectionError: If detection fails (model loading, inference error)

    Implementation:
        1. Load model checkpoint (cached in module-level singleton)
        2. Create ComponentClassifierInference
        3. Call predict_image_layout(image)
        4. Return result dict
        5. On error: raise LayoutDetectionError with context
    """
    global _inference_engine, _model_path

    try:
        # Lazy-load and cache inference engine
        if _inference_engine is None or _model_path != model_path:
            _model_path = model_path
            _inference_engine = ComponentClassifierInference(model_path)

        # Perform inference
        result = _inference_engine.predict_image_layout(image)

        return result

    except FileNotFoundError as e:
        # Model file not found
        raise LayoutDetectionError(
            f"ML layout detector: model not found at {model_path}. {str(e)}"
        ) from e
    except Exception as e:
        # Any other error during inference
        raise LayoutDetectionError(
            f"ML layout detector failed: {str(e)}"
        ) from e


def _use_brightness_fallback() -> None:
    """Reset module-level cache to force brightness-based detection."""
    global _inference_engine, _model_path
    _inference_engine = None
    _model_path = None
