"""Unit tests for ModelTrainer helper methods."""

import pytest
import numpy as np
import torch
from ane_design_model.model_trainer import ModelTrainer
from ane_design_model import PATCH_SIZE


def test_region_dict_to_patch_labels_with_known_regions():
    """Test _region_dict_to_patch_labels with synthetic region dict."""
    trainer = ModelTrainer()

    # Create a simple 300x400 region dict (typical synthetic image)
    region_dict = {
        "header": {"x": 0, "y": 0, "width": 400, "height": 45},
        "sidebar": {"x": 0, "y": 45, "width": 100, "height": 210},
        "content": {"x": 100, "y": 45, "width": 300, "height": 210},
        "footer": {"x": 0, "y": 255, "width": 400, "height": 45}
    }

    image_shape = (300, 400, 3)
    patch_labels = trainer._region_dict_to_patch_labels(region_dict, image_shape)

    # Verify patch_labels is a dict with integer keys and values
    assert isinstance(patch_labels, dict)
    assert all(isinstance(k, int) for k in patch_labels.keys())
    assert all(isinstance(v, int) for v in patch_labels.values())

    # For 300x400 image with 128x128 patches: 3 rows x 4 cols = 12 patches
    # (300 / 128 = 2.34 → 3, 400 / 128 = 3.125 → 4)
    patches_h = (300 + PATCH_SIZE - 1) // PATCH_SIZE  # 3
    patches_w = (400 + PATCH_SIZE - 1) // PATCH_SIZE  # 4
    expected_count = patches_h * patches_w

    assert len(patch_labels) == expected_count

    # Verify all labels are in valid range (0-3)
    for label in patch_labels.values():
        assert 0 <= label <= 3


def test_extract_patches_shape_and_dtype():
    """Test _extract_patches_and_labels returns correct shapes and dtypes."""
    trainer = ModelTrainer()

    # Create a simple test image (300x400x3)
    image = np.random.randint(0, 256, (300, 400, 3), dtype=np.uint8)

    # Create simple patch labels (12 patches for 300x400 image)
    patches_h = (300 + PATCH_SIZE - 1) // PATCH_SIZE
    patches_w = (400 + PATCH_SIZE - 1) // PATCH_SIZE
    patch_count = patches_h * patches_w
    patch_labels = {i: i % 4 for i in range(patch_count)}  # Cycle through 0-3

    patches, labels = trainer._extract_patches_and_labels(image, patch_labels)

    # Verify shapes
    assert patches.shape == (patch_count, 3, 128, 128), f"Expected ({patch_count}, 3, 128, 128), got {patches.shape}"
    assert labels.shape == (patch_count,), f"Expected ({patch_count},), got {labels.shape}"

    # Verify dtypes
    assert patches.dtype == torch.float32
    assert labels.dtype == torch.int64


def test_patch_normalization_range():
    """Test patches are normalized to [0, 1] after extraction."""
    trainer = ModelTrainer()

    # Create test image with known values
    image = np.zeros((128, 128, 3), dtype=np.uint8)
    image[:, :64, :] = 0    # Left half: black (0)
    image[:, 64:, :] = 255  # Right half: white (255)

    patch_labels = {0: 0}  # Single patch

    patches, labels = trainer._extract_patches_and_labels(image, patch_labels)

    # Verify normalization: min should be ~0.0, max should be ~1.0
    assert patches.min().item() >= 0.0
    assert patches.max().item() <= 1.0

    # Check specific values
    assert torch.isclose(patches[0, :, :, :64].min(), torch.tensor(0.0))
    assert torch.isclose(patches[0, :, :, 64:].max(), torch.tensor(1.0))


def test_edge_patches_padding():
    """Test that edge patches are zero-padded correctly."""
    trainer = ModelTrainer()

    # Create a small image that doesn't divide evenly into 128x128
    # e.g., 200x200 → needs 2x2 grid, but edge patches are only 72x72
    image = np.ones((200, 200, 3), dtype=np.uint8) * 128  # Mid-gray

    patches_h = (200 + PATCH_SIZE - 1) // PATCH_SIZE  # 2
    patches_w = (200 + PATCH_SIZE - 1) // PATCH_SIZE  # 2
    patch_count = patches_h * patches_w  # 4
    patch_labels = {i: 0 for i in range(patch_count)}

    patches, labels = trainer._extract_patches_and_labels(image, patch_labels)

    # All patches should be 128x128
    assert patches.shape == (4, 3, 128, 128)

    # Edge patches should have some zero padding
    # Patch 0: [0:128, 0:128] - covers [0:128, 0:128] of image (full)
    # Patch 1: [0:128, 128:256] - covers [0:128, 128:200] of image (72px wide, padded)
    # Patch 2: [128:256, 0:128] - covers [128:200, 0:128] of image (72px tall, padded)
    # Patch 3: [128:256, 128:256] - covers [128:200, 128:200] (72x72, padded)

    # Check patch 3 (bottom-right) has padding
    patch_3 = patches[3]
    # The actual image region is 72x72, rest should be zero-padded
    # Check that padding region (beyond 72x72) is near zero
    padding_region = patch_3[:, 72:, :]  # Bottom padding
    assert padding_region.max().item() < 0.01  # Should be ~0 from padding


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
