"""ANE Design Model Package - Shared types, constants, and utility functions.

This module defines all cross-component types, constants, and configuration defaults
for the neural design-to-HTML component classifier.
"""

from typing import TypeAlias

# ============================================================================
# Component Class Mappings
# ============================================================================

COMPONENT_CLASSES = {
    "header": 0,      # Top region
    "nav": 1,         # Sidebar / navigation region
    "card": 2,        # Content / main region
    "footer": 3       # Bottom region
}

REVERSE_COMPONENT_CLASSES = {
    0: "header",
    1: "nav",
    2: "card",
    3: "footer"
}

# ============================================================================
# Image and Patch Specifications
# ============================================================================

PATCH_SIZE = 128                    # 128×128 pixel patches for model input
SYNTHETIC_IMAGE_HEIGHT = 300        # Synthetic design image height
SYNTHETIC_IMAGE_WIDTH = 400         # Synthetic design image width
SYNTHETIC_IMAGE_CHANNELS = 3        # RGB

# ============================================================================
# Training Configuration
# ============================================================================

DEFAULT_BATCH_SIZE = 16
DEFAULT_EPOCHS = 10
DEFAULT_LEARNING_RATE = 0.001
DEFAULT_TRAIN_VAL_SPLIT = 0.8

# ============================================================================
# Model Architecture Parameters
# ============================================================================

MODEL_INPUT_CHANNELS = 3
MODEL_INPUT_SIZE = 128              # H=W=128 for patches
MODEL_NUM_CLASSES = 4               # header, nav, card, footer
MODEL_HIDDEN_DIM = 128              # MLP hidden layer size
MODEL_MAX_SIZE_BYTES = 10485760     # 10 MB

# ============================================================================
# Type Aliases
# ============================================================================

ComponentDict: TypeAlias = dict[str, dict[str, int] | None]
"""Type alias for component region dictionary.

Example:
    {
        "header": {"x": 0, "y": 0, "width": 400, "height": 45},
        "nav": {"x": 0, "y": 45, "width": 100, "height": 255},
        "card": {"x": 100, "y": 45, "width": 300, "height": 255},
        "footer": {"x": 0, "y": 255, "width": 400, "height": 45}
    }

The dict maps region names to their bounding box info (x, y, width, height) or None
if that region is not detected.
"""

LabelsDict: TypeAlias = dict[str, ComponentDict]
"""Type alias for dataset labels dictionary.

Maps image filenames to their component regions.

Example:
    {
        "img_001.png": {"header": {...}, "nav": {...}, ...},
        "img_002.png": {"header": {...}, "nav": {...}, ...},
        ...
    }
"""

RegionInfo: TypeAlias = dict[str, int | None]
"""Type alias for individual region information.

Example:
    {"x": 0, "y": 0, "width": 400, "height": 45}

Fields: x, y (top-left corner), width, height (dimensions)
"""

# ============================================================================
# Utility Functions
# ============================================================================


def get_component_class(region_name: str) -> int:
    """Get numeric class ID for region name.

    Maps region names (including aliases) to their class IDs.

    Args:
        region_name: Region name string. Accepts:
            - "header" → 0
            - "nav", "sidebar" → 1
            - "card", "content" → 2
            - "footer" → 3

    Returns:
        Integer class ID (0-3)

    Raises:
        ValueError: If region_name is not recognized
    """
    # Normalize aliases
    normalized = {
        "header": "header",
        "nav": "nav",
        "sidebar": "nav",  # alias
        "card": "card",
        "content": "card",  # alias
        "footer": "footer"
    }

    if region_name not in normalized:
        valid_names = ", ".join(sorted(set(normalized.keys())))
        raise ValueError(
            f"Unknown region name: {region_name!r}. "
            f"Valid names: {valid_names}"
        )

    canonical_name = normalized[region_name]
    return COMPONENT_CLASSES[canonical_name]


def get_region_name(class_id: int) -> str:
    """Get region name for numeric class ID.

    Maps class IDs back to canonical region names.

    Args:
        class_id: Integer class ID (0-3)

    Returns:
        Canonical region name string:
            - 0 → "header"
            - 1 → "nav"
            - 2 → "card"
            - 3 → "footer"

    Raises:
        ValueError: If class_id is out of valid range (0-3)
    """
    if class_id not in REVERSE_COMPONENT_CLASSES:
        raise ValueError(
            f"Invalid class ID: {class_id}. "
            f"Valid class IDs are 0-3 (header, nav, card, footer)"
        )

    return REVERSE_COMPONENT_CLASSES[class_id]


__all__ = [
    # Constants - Component mapping
    "COMPONENT_CLASSES",
    "REVERSE_COMPONENT_CLASSES",
    # Constants - Image/patch dimensions
    "PATCH_SIZE",
    "SYNTHETIC_IMAGE_HEIGHT",
    "SYNTHETIC_IMAGE_WIDTH",
    "SYNTHETIC_IMAGE_CHANNELS",
    # Constants - Training configuration
    "DEFAULT_BATCH_SIZE",
    "DEFAULT_EPOCHS",
    "DEFAULT_LEARNING_RATE",
    "DEFAULT_TRAIN_VAL_SPLIT",
    # Constants - Model architecture
    "MODEL_INPUT_CHANNELS",
    "MODEL_INPUT_SIZE",
    "MODEL_NUM_CLASSES",
    "MODEL_HIDDEN_DIM",
    "MODEL_MAX_SIZE_BYTES",
    # Type aliases
    "ComponentDict",
    "LabelsDict",
    "RegionInfo",
    # Functions
    "get_component_class",
    "get_region_name",
]
