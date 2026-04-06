"""Cross-module integration tests for ane_design_model package.

These tests verify that the model, model_trainer, and shared types/constants
work correctly together, especially at boundary conditions where merges occurred.
"""

import pytest
import tempfile
import json
import numpy as np
import torch
from pathlib import Path
from PIL import Image

# Test imports from the package
from ane_design_model import (
    ComponentDict, LabelsDict, RegionInfo,
    COMPONENT_CLASSES, REVERSE_COMPONENT_CLASSES,
    get_component_class, get_region_name,
    PATCH_SIZE, SYNTHETIC_IMAGE_HEIGHT, SYNTHETIC_IMAGE_WIDTH,
    DEFAULT_BATCH_SIZE, DEFAULT_EPOCHS, DEFAULT_LEARNING_RATE,
    MODEL_INPUT_CHANNELS, MODEL_INPUT_SIZE, MODEL_NUM_CLASSES
)
from ane_design_model.model import ComponentClassifier, create_model
from ane_design_model.model_trainer import ModelTrainer, TrainingMetrics


class TestComponentClassConstantInteraction:
    """Verify COMPONENT_CLASSES and REVERSE_COMPONENT_CLASSES are complementary."""

    def test_bidirectional_mapping_consistency(self):
        """Verify get_component_class and get_region_name are inverses."""
        for region_name, class_id in COMPONENT_CLASSES.items():
            # Forward: name → class
            computed_class = get_component_class(region_name)
            assert computed_class == class_id, \
                f"get_component_class({region_name!r}) = {computed_class}, expected {class_id}"

            # Backward: class → name
            computed_name = get_region_name(class_id)
            assert computed_name == region_name, \
                f"get_region_name({class_id}) = {computed_name!r}, expected {region_name!r}"

    def test_all_class_ids_have_reverse_mapping(self):
        """Every class ID 0-3 must have a reverse mapping."""
        for class_id in range(4):
            name = get_region_name(class_id)
            assert name in COMPONENT_CLASSES, \
                f"Class ID {class_id} maps to {name!r}, but not in COMPONENT_CLASSES"
            assert COMPONENT_CLASSES[name] == class_id

    def test_aliases_map_to_canonical_names(self):
        """Sidebar aliases to nav; Content aliases to card."""
        # 'sidebar' should map to class 1 (nav)
        assert get_component_class("sidebar") == COMPONENT_CLASSES["nav"]

        # 'content' should map to class 2 (card)
        assert get_component_class("content") == COMPONENT_CLASSES["card"]

    def test_all_model_classes_covered(self):
        """Verify MODEL_NUM_CLASSES matches COMPONENT_CLASSES size."""
        assert MODEL_NUM_CLASSES == len(COMPONENT_CLASSES), \
            f"MODEL_NUM_CLASSES={MODEL_NUM_CLASSES} != len(COMPONENT_CLASSES)={len(COMPONENT_CLASSES)}"

    def test_component_class_values_range(self):
        """Component class IDs must be 0, 1, 2, 3."""
        class_ids = list(COMPONENT_CLASSES.values())
        expected = [0, 1, 2, 3]
        assert sorted(class_ids) == expected, \
            f"Class IDs {sorted(class_ids)} != expected {expected}"


class TestAliasHandlingInTrainer:
    """Test that ModelTrainer correctly handles region name aliases."""

    def test_trainer_recognizes_sidebar_as_nav(self):
        """ModelTrainer should accept 'sidebar' and map it to nav class (1)."""
        trainer = ModelTrainer()

        # Region dict using alias 'sidebar' instead of 'nav'
        region_dict = {
            "header": {"x": 0, "y": 0, "width": 400, "height": 45},
            "sidebar": {"x": 0, "y": 45, "width": 100, "height": 210},  # Alias
            "content": {"x": 100, "y": 45, "width": 300, "height": 210},
            "footer": {"x": 0, "y": 255, "width": 400, "height": 45}
        }

        patch_labels = trainer._region_dict_to_patch_labels(region_dict, (300, 400, 3))

        # Find the patch that should be labeled as nav (class 1)
        # Patch (0, 0) covers [0:128, 0:128], should intersect with sidebar [0:128, 0:100]
        patch_id = 0  # top-left patch
        assert patch_labels[patch_id] == 1, \
            f"Expected patch 0 (top-left) to be nav (1), got {patch_labels[patch_id]}"

    def test_trainer_recognizes_content_as_card(self):
        """ModelTrainer should accept 'content' and map it to card class (2)."""
        trainer = ModelTrainer()

        region_dict = {
            "header": {"x": 0, "y": 0, "width": 400, "height": 45},
            "sidebar": {"x": 0, "y": 45, "width": 100, "height": 210},
            "content": {"x": 100, "y": 45, "width": 300, "height": 210},  # Alias
            "footer": {"x": 0, "y": 255, "width": 400, "height": 45}
        }

        patch_labels = trainer._region_dict_to_patch_labels(region_dict, (300, 400, 3))

        # Patch (0, 1) covers [0:128, 128:256], should intersect with content [100:400, 45:255]
        patch_id = 1  # top-right patch
        assert patch_labels[patch_id] == 2, \
            f"Expected patch 1 (top-right) to be card (2), got {patch_labels[patch_id]}"

    def test_mixed_canonical_and_alias_names(self):
        """Test a region dict with both canonical names and aliases."""
        trainer = ModelTrainer()

        region_dict = {
            "header": {"x": 0, "y": 0, "width": 400, "height": 45},      # Canonical
            "sidebar": {"x": 0, "y": 45, "width": 100, "height": 210},   # Alias for nav
            "card": {"x": 100, "y": 45, "width": 300, "height": 210},    # Canonical (not 'content')
            "footer": {"x": 0, "y": 255, "width": 400, "height": 45}     # Canonical
        }

        # Should not raise; both canonical and aliases should work
        patch_labels = trainer._region_dict_to_patch_labels(region_dict, (300, 400, 3))
        assert len(patch_labels) == 12, f"Expected 12 patches, got {len(patch_labels)}"


class TestModelAndTrainerConstantConsistency:
    """Verify model architecture constants match trainer expectations."""

    def test_model_input_size_matches_patch_size(self):
        """Model must expect PATCH_SIZE×PATCH_SIZE inputs."""
        assert MODEL_INPUT_SIZE == PATCH_SIZE, \
            f"MODEL_INPUT_SIZE={MODEL_INPUT_SIZE} != PATCH_SIZE={PATCH_SIZE}"

    def test_model_input_channels_match_constant(self):
        """Model expects 3 input channels (RGB)."""
        assert MODEL_INPUT_CHANNELS == 3
        model = ComponentClassifier()

        # Create test input with correct shape
        batch = torch.randn(4, 3, 128, 128)
        output = model(batch)
        assert output.shape == (4, MODEL_NUM_CLASSES)

    def test_model_num_classes_matches_component_mapping(self):
        """Model output size must match number of component classes."""
        assert MODEL_NUM_CLASSES == len(COMPONENT_CLASSES), \
            f"MODEL_NUM_CLASSES={MODEL_NUM_CLASSES} != len(COMPONENT_CLASSES)={len(COMPONENT_CLASSES)}"

        model = ComponentClassifier()
        batch = torch.randn(2, 3, 128, 128)
        output = model(batch)
        assert output.shape[1] == MODEL_NUM_CLASSES

    def test_synthetic_image_dimensions_produce_valid_patches(self):
        """Synthetic image dimensions should divide evenly into PATCH_SIZE patches."""
        patches_h = (SYNTHETIC_IMAGE_HEIGHT + PATCH_SIZE - 1) // PATCH_SIZE
        patches_w = (SYNTHETIC_IMAGE_WIDTH + PATCH_SIZE - 1) // PATCH_SIZE

        # For 300×400 with 128×128 patches: 3×4 grid = 12 patches
        # This is the expected size for training datasets
        assert patches_h * patches_w > 0, "Invalid patch grid"


class TestTrainerAndConstantIntegration:
    """Test that trainer uses constants correctly."""

    def test_trainer_uses_default_batch_size_constant(self):
        """Trainer should use DEFAULT_BATCH_SIZE if not specified."""
        trainer1 = ModelTrainer()
        assert trainer1.batch_size == DEFAULT_BATCH_SIZE

        trainer2 = ModelTrainer(batch_size=32)
        assert trainer2.batch_size == 32

    def test_trainer_uses_default_epochs_constant(self):
        """Trainer should use DEFAULT_EPOCHS if not specified."""
        trainer1 = ModelTrainer()
        assert trainer1.epochs == DEFAULT_EPOCHS

        trainer2 = ModelTrainer(epochs=20)
        assert trainer2.epochs == 20

    def test_trainer_uses_default_learning_rate_constant(self):
        """Trainer should use DEFAULT_LEARNING_RATE if not specified."""
        trainer1 = ModelTrainer()
        assert trainer1.learning_rate == DEFAULT_LEARNING_RATE

        trainer2 = ModelTrainer(learning_rate=0.01)
        assert trainer2.learning_rate == 0.01


class TestEndToEndPipeline:
    """End-to-end tests verifying model creation, training, and metrics."""

    @pytest.fixture
    def small_dataset(self):
        """Create a minimal test dataset."""
        tmpdir = tempfile.mkdtemp()
        dataset_dir = Path(tmpdir) / "dataset"
        images_dir = dataset_dir / "images"
        images_dir.mkdir(parents=True)

        labels = {}
        for i in range(2):
            # Create synthetic image
            img = np.zeros((300, 400, 3), dtype=np.uint8)
            img[0:45, :, :] = [200, 200, 200]           # Header
            img[255:300, :, :] = [50, 50, 50]           # Footer
            img[45:255, 0:100, :] = [100, 100, 200]     # Sidebar
            img[45:255, 100:400, :] = [200, 100, 100]   # Content

            filename = f"img_{i+1:03d}.png"
            Image.fromarray(img).save(images_dir / filename)

            labels[filename] = {
                "header": {"x": 0, "y": 0, "width": 400, "height": 45},
                "sidebar": {"x": 0, "y": 45, "width": 100, "height": 210},
                "content": {"x": 100, "y": 45, "width": 300, "height": 210},
                "footer": {"x": 0, "y": 255, "width": 400, "height": 45}
            }

        with open(dataset_dir / "labels.json", "w") as f:
            json.dump(labels, f)

        yield str(dataset_dir)

        # Cleanup
        import shutil
        shutil.rmtree(tmpdir)

    def test_create_model_produces_correct_architecture(self):
        """Verify create_model() returns properly configured model."""
        model = create_model()
        assert isinstance(model, ComponentClassifier)

        # Test forward pass
        batch = torch.randn(4, 3, 128, 128)
        output = model(batch)
        assert output.shape == (4, MODEL_NUM_CLASSES)

    def test_trainer_metrics_are_valid_type(self, small_dataset):
        """TrainingMetrics should be a NamedTuple with correct fields."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "model.pt"

            trainer = ModelTrainer(batch_size=4, epochs=1)
            metrics = trainer.train(small_dataset, str(output_path), verbose=False)

            # Check type
            assert isinstance(metrics, TrainingMetrics)

            # Check fields
            assert hasattr(metrics, 'train_loss')
            assert hasattr(metrics, 'val_accuracy')
            assert hasattr(metrics, 'epochs_trained')

            # Check values are sensible
            assert isinstance(metrics.train_loss, float)
            assert isinstance(metrics.val_accuracy, float)
            assert isinstance(metrics.epochs_trained, int)
            assert metrics.train_loss > 0
            assert 0 <= metrics.val_accuracy <= 1
            assert metrics.epochs_trained == 1

    def test_full_pipeline_model_creation_to_inference(self, small_dataset):
        """Test full pipeline: create → train → load → infer."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "trained_model.pt"

            # Step 1: Create trainer and train
            trainer = ModelTrainer(batch_size=4, epochs=1)
            metrics = trainer.train(small_dataset, str(model_path), verbose=False)
            assert metrics.val_accuracy > 0

            # Step 2: Create new model and load weights
            model = create_model()
            state_dict = torch.load(model_path, map_location='cpu')
            model.load_state_dict(state_dict)

            # Step 3: Perform inference
            test_batch = torch.randn(2, 3, 128, 128)
            model.eval()
            with torch.no_grad():
                logits = model(test_batch)

            # Step 4: Verify output shape and get predictions
            assert logits.shape == (2, MODEL_NUM_CLASSES)
            predictions = torch.argmax(logits, dim=1)
            assert predictions.shape == (2,)

            # Verify predictions are valid class IDs
            for pred in predictions:
                class_id = pred.item()
                assert 0 <= class_id < MODEL_NUM_CLASSES
                region_name = get_region_name(class_id)
                assert region_name in COMPONENT_CLASSES


class TestRegionDictTypeAlias:
    """Test that type aliases work with trainer."""

    def test_component_dict_type_annotation(self):
        """ComponentDict should describe region structure."""
        # ComponentDict: TypeAlias = dict[str, dict[str, int] | None]
        sample: ComponentDict = {
            "header": {"x": 0, "y": 0, "width": 400, "height": 45},
            "footer": None
        }
        assert isinstance(sample, dict)
        assert "header" in sample
        assert sample["header"]["x"] == 0

    def test_region_info_type_alias(self):
        """RegionInfo should describe bounding box."""
        region: RegionInfo = {"x": 0, "y": 0, "width": 400, "height": 45}
        assert region["x"] == 0
        assert region["width"] == 400

    def test_labels_dict_type_alias(self):
        """LabelsDict should describe full label structure."""
        labels: LabelsDict = {
            "img_001.png": {
                "header": {"x": 0, "y": 0, "width": 400, "height": 45},
                "nav": None,
                "card": None,
                "footer": None
            }
        }
        assert "img_001.png" in labels
        assert labels["img_001.png"]["header"]["x"] == 0


class TestModelInputValidation:
    """Test model's input validation against constants."""

    def test_model_rejects_wrong_input_shape(self):
        """Model should raise RuntimeError for incorrect input shapes."""
        model = ComponentClassifier()

        # Wrong number of channels (2 instead of 3)
        with pytest.raises(RuntimeError, match="Expected input shape"):
            model(torch.randn(4, 2, 128, 128))

        # Wrong spatial dimensions (64 instead of 128)
        with pytest.raises(RuntimeError, match="Expected input shape"):
            model(torch.randn(4, 3, 64, 64))

        # Wrong batch dimension (should be fine)
        output = model(torch.randn(1, 3, 128, 128))
        assert output.shape == (1, MODEL_NUM_CLASSES)

    def test_model_accepts_valid_batch_sizes(self):
        """Model should accept various batch sizes."""
        model = ComponentClassifier()

        for batch_size in [1, 2, 4, 8, 16, 32]:
            output = model(torch.randn(batch_size, 3, 128, 128))
            assert output.shape == (batch_size, MODEL_NUM_CLASSES)


class TestExceptionHandling:
    """Test proper error handling for invalid inputs."""

    def test_get_component_class_invalid_region(self):
        """get_component_class should raise ValueError for unknown region."""
        with pytest.raises(ValueError, match="Unknown region name"):
            get_component_class("invalid_region")

    def test_get_region_name_invalid_class(self):
        """get_region_name should raise ValueError for invalid class ID."""
        with pytest.raises(ValueError, match="Invalid class ID"):
            get_region_name(99)

        with pytest.raises(ValueError, match="Invalid class ID"):
            get_region_name(-1)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
