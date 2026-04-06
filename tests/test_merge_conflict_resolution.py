"""Tests for merge conflict resolution areas.

These tests specifically exercise code paths where conflicts were resolved,
particularly focusing on region name alias handling and patch label conversion.
"""

import pytest
import tempfile
import json
import numpy as np
import torch
from pathlib import Path
from PIL import Image

from ane_design_model import (
    COMPONENT_CLASSES, PATCH_SIZE,
    get_component_class, get_region_name
)
from ane_design_model.model import ComponentClassifier
from ane_design_model.model_trainer import ModelTrainer


class TestRegionNameNormalization:
    """Test region name normalization - a key conflict resolution area.

    The _region_dict_to_patch_labels method has inline alias handling:
    - "sidebar" → "nav" (class 1)
    - "content" → "card" (class 2)

    This tests that the normalization is correct and consistent with get_component_class.
    """

    def test_sidebar_alias_in_region_dict_maps_to_nav_class(self):
        """Test that 'sidebar' in region_dict maps to nav class (1)."""
        trainer = ModelTrainer()

        # Create region dict with 'sidebar' alias
        region_dict = {
            "header": {"x": 0, "y": 0, "width": 400, "height": 45},
            "sidebar": {"x": 0, "y": 45, "width": 100, "height": 210},  # Alias
            "content": {"x": 100, "y": 45, "width": 300, "height": 210},
            "footer": {"x": 0, "y": 255, "width": 400, "height": 45}
        }

        patch_labels = trainer._region_dict_to_patch_labels(region_dict, (300, 400, 3))

        # Verify class mapping for sidebar-affected patches
        # Get patches that intersect with sidebar region [0:100, 45:255]
        sidebar_patch_ids = []
        for patch_id, label in patch_labels.items():
            if label == 1:  # nav class
                sidebar_patch_ids.append(patch_id)

        # Should have at least one patch labeled as nav
        assert len(sidebar_patch_ids) > 0, \
            "Expected at least one patch to be classified as nav (class 1)"

    def test_content_alias_in_region_dict_maps_to_card_class(self):
        """Test that 'content' in region_dict maps to card class (2)."""
        trainer = ModelTrainer()

        region_dict = {
            "header": {"x": 0, "y": 0, "width": 400, "height": 45},
            "sidebar": {"x": 0, "y": 45, "width": 100, "height": 210},
            "content": {"x": 100, "y": 45, "width": 300, "height": 210},  # Alias
            "footer": {"x": 0, "y": 255, "width": 400, "height": 45}
        }

        patch_labels = trainer._region_dict_to_patch_labels(region_dict, (300, 400, 3))

        # Get patches labeled as card (class 2)
        card_patch_ids = [pid for pid, label in patch_labels.items() if label == 2]

        # Should have at least one patch labeled as card
        assert len(card_patch_ids) > 0, \
            "Expected at least one patch to be classified as card (class 2)"

    def test_conflict_resolution_sidebar_vs_nav_consistency(self):
        """Verify sidebar handling is consistent with get_component_class('nav')."""
        trainer = ModelTrainer()

        # Both dicts should produce the same patch labels due to alias handling
        region_dict_sidebar = {
            "header": {"x": 0, "y": 0, "width": 400, "height": 45},
            "sidebar": {"x": 0, "y": 45, "width": 100, "height": 210},  # Use 'sidebar'
            "content": {"x": 100, "y": 45, "width": 300, "height": 210},
            "footer": {"x": 0, "y": 255, "width": 400, "height": 45}
        }

        region_dict_nav = {
            "header": {"x": 0, "y": 0, "width": 400, "height": 45},
            "nav": {"x": 0, "y": 45, "width": 100, "height": 210},  # Use 'nav' instead
            "card": {"x": 100, "y": 45, "width": 300, "height": 210},
            "footer": {"x": 0, "y": 255, "width": 400, "height": 45}
        }

        labels_sidebar = trainer._region_dict_to_patch_labels(
            region_dict_sidebar, (300, 400, 3)
        )
        labels_nav = trainer._region_dict_to_patch_labels(
            region_dict_nav, (300, 400, 3)
        )

        # Both should produce the same patch labeling
        assert labels_sidebar == labels_nav, \
            "Patch labels differ between 'sidebar' and 'nav' region dicts"

    def test_conflict_resolution_content_vs_card_consistency(self):
        """Verify content handling is consistent with get_component_class('card')."""
        trainer = ModelTrainer()

        region_dict_content = {
            "header": {"x": 0, "y": 0, "width": 400, "height": 45},
            "sidebar": {"x": 0, "y": 45, "width": 100, "height": 210},
            "content": {"x": 100, "y": 45, "width": 300, "height": 210},  # Use 'content'
            "footer": {"x": 0, "y": 255, "width": 400, "height": 45}
        }

        region_dict_card = {
            "header": {"x": 0, "y": 0, "width": 400, "height": 45},
            "nav": {"x": 0, "y": 45, "width": 100, "height": 210},
            "card": {"x": 100, "y": 45, "width": 300, "height": 210},  # Use 'card'
            "footer": {"x": 0, "y": 255, "width": 400, "height": 45}
        }

        labels_content = trainer._region_dict_to_patch_labels(
            region_dict_content, (300, 400, 3)
        )
        labels_card = trainer._region_dict_to_patch_labels(
            region_dict_card, (300, 400, 3)
        )

        # Both should produce the same patch labeling
        assert labels_content == labels_card, \
            "Patch labels differ between 'content' and 'card' region dicts"

    def test_all_aliases_in_single_region_dict(self):
        """Test a region dict using all aliases simultaneously."""
        trainer = ModelTrainer()

        region_dict = {
            "header": {"x": 0, "y": 0, "width": 400, "height": 45},
            "sidebar": {"x": 0, "y": 45, "width": 100, "height": 210},   # Alias
            "content": {"x": 100, "y": 45, "width": 300, "height": 210},  # Alias
            "footer": {"x": 0, "y": 255, "width": 400, "height": 45}
        }

        # Should not raise; both aliases should be recognized
        patch_labels = trainer._region_dict_to_patch_labels(region_dict, (300, 400, 3))
        assert isinstance(patch_labels, dict)
        assert len(patch_labels) > 0


class TestMaxIntersectionAreaLogic:
    """Test the max intersection area labeling strategy.

    When a patch overlaps multiple regions, the patch gets the label of the region
    with the largest intersection area.
    """

    def test_patch_gets_label_of_largest_intersecting_region(self):
        """Verify patch is labeled with the region having max intersection."""
        trainer = ModelTrainer()

        # Create regions with clear max intersection
        region_dict = {
            "header": {"x": 0, "y": 0, "width": 400, "height": 45},
            "nav": {"x": 0, "y": 45, "width": 100, "height": 210},
            "card": {"x": 100, "y": 45, "width": 300, "height": 210},
            "footer": {"x": 0, "y": 255, "width": 400, "height": 45}
        }

        patch_labels = trainer._region_dict_to_patch_labels(region_dict, (300, 400, 3))

        # Patch 0 (top-left) at [0:128, 0:128] should have largest intersection with nav [0:100, 45:255]
        # nav intersection: [0:100, 45:128] = 100 × 83 = 8300
        # header intersection: [0:128, 0:45] = 128 × 45 = 5760
        # So patch 0 should be labeled nav (class 1)
        patch_id_0_0 = 0
        assert patch_labels[patch_id_0_0] == 1, \
            f"Expected patch (0,0) to be nav (1), got {patch_labels[patch_id_0_0]}"

    def test_default_label_when_no_regions_overlap(self):
        """When no regions overlap a patch, it defaults to card (class 2)."""
        trainer = ModelTrainer()

        # Create regions that don't cover the bottom-right corner
        region_dict = {
            "header": {"x": 0, "y": 0, "width": 400, "height": 50},
            "nav": None,
            "card": None,
            "footer": None
        }

        patch_labels = trainer._region_dict_to_patch_labels(region_dict, (300, 400, 3))

        # Patches below the header should default to card (class 2)
        # E.g., patch (1, 0) at [128:256, 0:128] should be card
        patch_id_1_0 = 4  # Row 1, Col 0 (3 cols per row, so row 1 starts at patch 4)
        assert patch_labels[patch_id_1_0] == 2, \
            f"Expected patch (1,0) to default to card (2), got {patch_labels[patch_id_1_0]}"

    def test_none_regions_are_skipped(self):
        """Regions with None value should be skipped in intersection logic."""
        trainer = ModelTrainer()

        region_dict = {
            "header": {"x": 0, "y": 0, "width": 400, "height": 45},
            "nav": None,  # Not present
            "card": {"x": 100, "y": 45, "width": 300, "height": 210},
            "footer": {"x": 0, "y": 255, "width": 400, "height": 45}
        }

        # Should not raise due to None
        patch_labels = trainer._region_dict_to_patch_labels(region_dict, (300, 400, 3))
        assert isinstance(patch_labels, dict)


class TestPatchExtractionWithNormalization:
    """Test patch extraction handles normalization correctly.

    Patches must be normalized to [0, 1] float32 for model input.
    """

    def test_extracted_patches_are_float32_in_range(self):
        """Extracted patches must be float32 in [0, 1]."""
        trainer = ModelTrainer()

        # Create test image with known values
        image = np.array([[[0, 128, 255]] * 400] * 300, dtype=np.uint8)
        patch_labels = {0: 0, 1: 1, 2: 2, 3: 3}

        patches, labels = trainer._extract_patches_and_labels(image, patch_labels)

        assert patches.dtype == torch.float32, f"Expected float32, got {patches.dtype}"
        assert patches.min().item() >= 0.0
        assert patches.max().item() <= 1.0

    def test_zero_pixel_becomes_zero_float(self):
        """Black pixels (0) should map to 0.0."""
        trainer = ModelTrainer()

        image = np.zeros((128, 128, 3), dtype=np.uint8)
        patch_labels = {0: 0}

        patches, _ = trainer._extract_patches_and_labels(image, patch_labels)

        # All pixels should be 0.0
        assert torch.allclose(patches[0], torch.tensor(0.0))

    def test_255_pixel_becomes_one_float(self):
        """White pixels (255) should map to 1.0."""
        trainer = ModelTrainer()

        image = np.full((128, 128, 3), 255, dtype=np.uint8)
        patch_labels = {0: 0}

        patches, _ = trainer._extract_patches_and_labels(image, patch_labels)

        # All pixels should be ~1.0
        assert torch.allclose(patches[0], torch.tensor(1.0))

    def test_mid_value_pixel_normalized_correctly(self):
        """Mid-range pixel value should normalize to ~0.5."""
        trainer = ModelTrainer()

        image = np.full((128, 128, 3), 128, dtype=np.uint8)
        patch_labels = {0: 0}

        patches, _ = trainer._extract_patches_and_labels(image, patch_labels)

        # 128/255 ≈ 0.502
        expected = 128.0 / 255.0
        assert torch.allclose(patches[0], torch.tensor(expected), atol=0.01)


class TestTrainingWithMixedRegionNames:
    """Test that training works correctly with mixed region naming."""

    @pytest.fixture
    def mixed_name_dataset(self):
        """Create dataset using both canonical and alias region names."""
        tmpdir = tempfile.mkdtemp()
        dataset_dir = Path(tmpdir) / "dataset"
        images_dir = dataset_dir / "images"
        images_dir.mkdir(parents=True)

        labels = {}

        # Image 1: Use aliases (sidebar, content)
        img1 = np.zeros((300, 400, 3), dtype=np.uint8)
        img1[0:45, :, :] = 200
        img1[45:255, 0:100, :] = 100
        img1[45:255, 100:400, :] = 150
        img1[255:300, :, :] = 50
        Image.fromarray(img1).save(images_dir / "img_001.png")
        labels["img_001.png"] = {
            "header": {"x": 0, "y": 0, "width": 400, "height": 45},
            "sidebar": {"x": 0, "y": 45, "width": 100, "height": 210},  # Alias
            "content": {"x": 100, "y": 45, "width": 300, "height": 210},  # Alias
            "footer": {"x": 0, "y": 255, "width": 400, "height": 45}
        }

        # Image 2: Use canonical names (nav, card)
        img2 = np.zeros((300, 400, 3), dtype=np.uint8)
        img2[0:45, :, :] = 200
        img2[45:255, 0:100, :] = 100
        img2[45:255, 100:400, :] = 150
        img2[255:300, :, :] = 50
        Image.fromarray(img2).save(images_dir / "img_002.png")
        labels["img_002.png"] = {
            "header": {"x": 0, "y": 0, "width": 400, "height": 45},
            "nav": {"x": 0, "y": 45, "width": 100, "height": 210},  # Canonical
            "card": {"x": 100, "y": 45, "width": 300, "height": 210},  # Canonical
            "footer": {"x": 0, "y": 255, "width": 400, "height": 45}
        }

        with open(dataset_dir / "labels.json", "w") as f:
            json.dump(labels, f)

        yield str(dataset_dir)

        import shutil
        shutil.rmtree(tmpdir)

    def test_trainer_handles_mixed_region_names_in_training(self, mixed_name_dataset):
        """Trainer should handle dataset with mixed region names."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "model.pt"

            trainer = ModelTrainer(batch_size=4, epochs=1)

            # Should not raise; should handle both aliases and canonical names
            metrics = trainer.train(mixed_name_dataset, str(output_path), verbose=False)

            assert metrics.val_accuracy > 0
            assert metrics.epochs_trained == 1
            assert output_path.exists()


class TestModelLoadingAfterConflictResolution:
    """Test that trained models can be loaded and used correctly."""

    def test_model_state_dict_contains_all_weights(self):
        """Saved model should have all required weight matrices."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model = ComponentClassifier()

            # Save and load
            model_path = Path(tmpdir) / "model.pt"
            torch.save(model.state_dict(), model_path)

            state = torch.load(model_path, map_location='cpu')

            # Verify all expected keys present
            expected_keys = {
                'conv1.weight', 'conv1.bias',
                'conv2.weight', 'conv2.bias',
                'fc1.weight', 'fc1.bias',
                'fc2.weight', 'fc2.bias'
            }

            actual_keys = set(state.keys())
            assert expected_keys == actual_keys, \
                f"Missing keys: {expected_keys - actual_keys}, Extra: {actual_keys - expected_keys}"

    def test_loaded_model_produces_consistent_output(self):
        """Model loaded from checkpoint should produce consistent predictions."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create and save model
            model1 = ComponentClassifier()
            model_path = Path(tmpdir) / "model.pt"
            torch.save(model1.state_dict(), model_path)

            # Load into new model
            model2 = ComponentClassifier()
            state = torch.load(model_path, map_location='cpu')
            model2.load_state_dict(state)

            # Both should produce same output on same input
            test_input = torch.randn(2, 3, 128, 128)

            model1.eval()
            model2.eval()

            with torch.no_grad():
                out1 = model1(test_input)
                out2 = model2(test_input)

            # Should be identical (they share the same initialized weights)
            assert torch.allclose(out1, out2, atol=1e-6)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
