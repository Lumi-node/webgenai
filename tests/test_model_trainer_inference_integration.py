"""Integration tests between model_trainer and inference modules.

Tests the integration between ModelTrainer (which produces trained models)
and ComponentClassifierInference (which consumes them), including:
1. Model checkpoint compatibility
2. Region label mapping consistency
3. Patch label generation vs inference logic alignment
4. Training-to-inference compatibility
"""

import pytest
import numpy as np
import torch
import tempfile
import json
from pathlib import Path
from PIL import Image

from ane_design_model.model_trainer import ModelTrainer
from ane_design_model.inference import ComponentClassifierInference
from ane_design_model import PATCH_SIZE, COMPONENT_CLASSES


class TestModelTrainerInferenceCompatibility:
    """Test compatibility between ModelTrainer output and ComponentClassifierInference input."""

    @pytest.fixture
    def synthetic_dataset(self):
        """Create a minimal synthetic dataset for training."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_dir = Path(tmpdir) / "dataset"
            images_dir = dataset_dir / "images"
            images_dir.mkdir(parents=True)

            # Create a few minimal images
            labels_data = {}
            for i in range(2):
                # Create synthetic image
                img = Image.new('RGB', (300, 400), color=(100 + i*20, 100 + i*20, 100 + i*20))
                img_path = images_dir / f"img_{i:03d}.png"
                img.save(img_path)

                # Create synthetic labels
                labels_data[f"img_{i:03d}.png"] = {
                    "header": {"x": 0, "y": 0, "width": 400, "height": 50},
                    "sidebar": {"x": 0, "y": 50, "width": 100, "height": 300},
                    "content": {"x": 100, "y": 50, "width": 300, "height": 300},
                    "footer": {"x": 0, "y": 350, "width": 400, "height": 50}
                }

            # Save labels
            labels_path = dataset_dir / "labels.json"
            with open(labels_path, "w") as f:
                json.dump(labels_data, f)

            yield str(dataset_dir)

    def test_trained_model_loads_in_inference_engine(self, synthetic_dataset):
        """Verify model trained by ModelTrainer can be loaded by ComponentClassifierInference."""
        # Train a model
        trainer = ModelTrainer(batch_size=1, epochs=1)
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "trained_model.pt"
            metrics = trainer.train(synthetic_dataset, str(model_path), verbose=False)

            # Model should be trained
            assert metrics.val_accuracy >= 0.0
            assert metrics.epochs_trained == 1

            # Model file should exist
            assert model_path.exists()

            # Should be loadable by inference engine
            inference_engine = ComponentClassifierInference(str(model_path))
            assert inference_engine is not None
            assert inference_engine.model is not None

    def test_inference_on_dataset_image(self, synthetic_dataset):
        """Verify inference engine can process images from the training dataset."""
        # Train a model
        trainer = ModelTrainer(batch_size=1, epochs=1)
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "trained_model.pt"
            trainer.train(synthetic_dataset, str(model_path), verbose=False)

            # Load inference engine
            inference_engine = ComponentClassifierInference(str(model_path))

            # Load an image from dataset
            dataset_path = Path(synthetic_dataset)
            test_image_path = list((dataset_path / "images").glob("*.png"))[0]
            test_image = Image.open(test_image_path).convert('RGB')
            test_image_array = np.array(test_image, dtype=np.uint8)

            # Should perform inference without error
            result = inference_engine.predict_image_layout(test_image_array)

            assert isinstance(result, dict)
            # predict_image_layout returns mapped names (sidebar, content not nav, card)
            assert all(k in result for k in ['header', 'sidebar', 'content', 'footer'])


class TestPatchLabelConsistency:
    """Test consistency between trainer's label generation and inference's patch processing."""

    @pytest.fixture
    def synthetic_dataset(self):
        """Create a minimal synthetic dataset."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_dir = Path(tmpdir) / "dataset"
            images_dir = dataset_dir / "images"
            images_dir.mkdir(parents=True)

            # Create a simple image
            img = Image.new('RGB', (PATCH_SIZE * 2, PATCH_SIZE * 2), color=(128, 128, 128))
            img_path = images_dir / "test.png"
            img.save(img_path)

            # Create labels with clear regions
            labels_data = {
                "test.png": {
                    "header": {"x": 0, "y": 0, "width": PATCH_SIZE * 2, "height": PATCH_SIZE},
                    "sidebar": {"x": 0, "y": PATCH_SIZE, "width": PATCH_SIZE, "height": PATCH_SIZE},
                    "content": {"x": PATCH_SIZE, "y": PATCH_SIZE, "width": PATCH_SIZE, "height": PATCH_SIZE},
                    "footer": None
                }
            }

            labels_path = dataset_dir / "labels.json"
            with open(labels_path, "w") as f:
                json.dump(labels_data, f)

            yield str(dataset_dir)

    def test_region_dict_to_patch_labels_mapping(self, synthetic_dataset):
        """Verify region-to-patch label mapping is consistent."""
        trainer = ModelTrainer(batch_size=1, epochs=1)

        # Get a sample image
        dataset_path = Path(synthetic_dataset)
        img_path = list((dataset_path / "images").glob("*.png"))[0]
        test_image = Image.open(img_path).convert('RGB')
        test_image_array = np.array(test_image, dtype=np.uint8)

        # Get region labels from dataset
        labels_path = dataset_path / "labels.json"
        with open(labels_path) as f:
            all_labels = json.load(f)
        region_dict = all_labels["test.png"]

        # Convert to patch labels using trainer's method
        patch_labels = trainer._region_dict_to_patch_labels(region_dict, test_image_array.shape)

        # Should have patch labels for all patches in grid
        patches_h = (test_image_array.shape[0] + PATCH_SIZE - 1) // PATCH_SIZE
        patches_w = (test_image_array.shape[1] + PATCH_SIZE - 1) // PATCH_SIZE
        expected_patch_count = patches_h * patches_w

        assert len(patch_labels) == expected_patch_count

        # Each label should be a valid class ID
        for patch_id, class_label in patch_labels.items():
            assert 0 <= class_label <= 3, f"Invalid class label {class_label} for patch {patch_id}"

    def test_extracted_patches_match_inference_input(self, synthetic_dataset):
        """Verify trainer's patch extraction matches inference's expectations."""
        trainer = ModelTrainer(batch_size=1, epochs=1)

        # Get a sample image
        dataset_path = Path(synthetic_dataset)
        img_path = list((dataset_path / "images").glob("*.png"))[0]
        test_image = Image.open(img_path).convert('RGB')
        test_image_array = np.array(test_image, dtype=np.uint8)

        # Get region labels
        labels_path = dataset_path / "labels.json"
        with open(labels_path) as f:
            all_labels = json.load(f)
        region_dict = all_labels["test.png"]

        # Extract patches using trainer's method
        patch_labels = trainer._region_dict_to_patch_labels(region_dict, test_image_array.shape)
        patches_tensor, labels_tensor = trainer._extract_patches_and_labels(
            test_image_array, patch_labels
        )

        # Verify patch format matches inference expectations
        assert patches_tensor.ndim == 4
        assert patches_tensor.shape[1] == 3  # 3 channels
        assert patches_tensor.shape[2] == PATCH_SIZE
        assert patches_tensor.shape[3] == PATCH_SIZE

        # Verify patches are in [0, 1] float range (trainer normalizes)
        assert patches_tensor.dtype == torch.float32
        assert patches_tensor.min() >= 0.0
        assert patches_tensor.max() <= 1.0

        # Verify labels match patch count
        assert labels_tensor.shape[0] == patches_tensor.shape[0]
        assert labels_tensor.dtype == torch.int64


class TestTrainerLabelAliasHandling:
    """Test that trainer correctly handles region name aliases (sidebar→nav, content→card)."""

    @pytest.fixture
    def synthetic_dataset(self):
        """Create dataset with region aliases."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_dir = Path(tmpdir) / "dataset"
            images_dir = dataset_dir / "images"
            images_dir.mkdir(parents=True)

            img = Image.new('RGB', (PATCH_SIZE * 2, PATCH_SIZE * 2), color=(128, 128, 128))
            img_path = images_dir / "test.png"
            img.save(img_path)

            # Create labels using output names (sidebar, content) instead of internal names
            labels_data = {
                "test.png": {
                    "header": {"x": 0, "y": 0, "width": PATCH_SIZE * 2, "height": PATCH_SIZE},
                    "sidebar": {"x": 0, "y": PATCH_SIZE, "width": PATCH_SIZE, "height": PATCH_SIZE},
                    "content": {"x": PATCH_SIZE, "y": PATCH_SIZE, "width": PATCH_SIZE, "height": PATCH_SIZE},
                    "footer": None
                }
            }

            labels_path = dataset_dir / "labels.json"
            with open(labels_path, "w") as f:
                json.dump(labels_data, f)

            yield str(dataset_dir)

    def test_trainer_maps_sidebar_to_nav(self, synthetic_dataset):
        """Verify trainer correctly maps 'sidebar' (output name) to 'nav' (internal class)."""
        trainer = ModelTrainer(batch_size=1, epochs=1)

        # Get image
        dataset_path = Path(synthetic_dataset)
        img_path = list((dataset_path / "images").glob("*.png"))[0]
        test_image = Image.open(img_path).convert('RGB')
        test_image_array = np.array(test_image, dtype=np.uint8)

        # Get labels
        labels_path = dataset_path / "labels.json"
        with open(labels_path) as f:
            all_labels = json.load(f)
        region_dict = all_labels["test.png"]

        # Convert to patch labels
        patch_labels = trainer._region_dict_to_patch_labels(region_dict, test_image_array.shape)

        # First patch (header) should be class 0
        assert patch_labels[0] == COMPONENT_CLASSES["header"]

        # Second and third patches (sidebar in first row, content in second row)
        # The exact patch IDs depend on grid layout, but sidebar should map to "nav" class
        assert COMPONENT_CLASSES["nav"] in patch_labels.values()

    def test_trainer_maps_content_to_card(self, synthetic_dataset):
        """Verify trainer correctly maps 'content' (output name) to 'card' (internal class)."""
        trainer = ModelTrainer(batch_size=1, epochs=1)

        # Get image
        dataset_path = Path(synthetic_dataset)
        img_path = list((dataset_path / "images").glob("*.png"))[0]
        test_image = Image.open(img_path).convert('RGB')
        test_image_array = np.array(test_image, dtype=np.uint8)

        # Get labels
        labels_path = dataset_path / "labels.json"
        with open(labels_path) as f:
            all_labels = json.load(f)
        region_dict = all_labels["test.png"]

        # Convert to patch labels
        patch_labels = trainer._region_dict_to_patch_labels(region_dict, test_image_array.shape)

        # Should contain card class
        assert COMPONENT_CLASSES["card"] in patch_labels.values()


class TestTrainerInferenceEndToEnd:
    """End-to-end tests training and then inferencing."""

    @pytest.fixture
    def synthetic_dataset(self):
        """Create a synthetic dataset."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_dir = Path(tmpdir) / "dataset"
            images_dir = dataset_dir / "images"
            images_dir.mkdir(parents=True)

            # Create multiple images
            labels_data = {}
            for i in range(3):
                img = Image.new('RGB', (300, 400), color=(100 + i*20, 100 + i*20, 100 + i*20))
                img_path = images_dir / f"img_{i:03d}.png"
                img.save(img_path)

                labels_data[f"img_{i:03d}.png"] = {
                    "header": {"x": 0, "y": 0, "width": 400, "height": 50},
                    "sidebar": {"x": 0, "y": 50, "width": 100, "height": 300},
                    "content": {"x": 100, "y": 50, "width": 300, "height": 300},
                    "footer": {"x": 0, "y": 350, "width": 400, "height": 50}
                }

            labels_path = dataset_dir / "labels.json"
            with open(labels_path, "w") as f:
                json.dump(labels_data, f)

            yield str(dataset_dir)

    def test_train_then_infer_same_image(self, synthetic_dataset):
        """Test training on dataset then inferencing on same image."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "model.pt"

            # Train
            trainer = ModelTrainer(batch_size=1, epochs=1)
            metrics = trainer.train(synthetic_dataset, str(model_path), verbose=False)
            assert metrics.val_accuracy >= 0.0

            # Load a training image
            dataset_path = Path(synthetic_dataset)
            test_image_path = list((dataset_path / "images").glob("*.png"))[0]
            test_image = Image.open(test_image_path).convert('RGB')
            test_image_array = np.array(test_image, dtype=np.uint8)

            # Infer
            inference_engine = ComponentClassifierInference(str(model_path))
            result = inference_engine.predict_image_layout(test_image_array)

            # Result should have expected keys (with output names)
            assert isinstance(result, dict)
            assert all(k in result for k in ['header', 'sidebar', 'content', 'footer'])

    def test_trained_model_produces_valid_logits(self, synthetic_dataset):
        """Verify trained model produces valid logits for inference."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "model.pt"

            # Train
            trainer = ModelTrainer(batch_size=1, epochs=1)
            trainer.train(synthetic_dataset, str(model_path), verbose=False)

            # Load and test inference on synthetic batch
            inference_engine = ComponentClassifierInference(str(model_path))

            # Create test patches
            test_patches = torch.randn(4, 3, PATCH_SIZE, PATCH_SIZE).to(inference_engine.device)
            test_patches = torch.clamp(test_patches, 0, 1)  # Ensure [0, 1] range

            # Get logits
            logits = inference_engine.predict_batch(test_patches)

            # Should have (batch, 4) shape
            assert logits.shape == (4, 4)

            # Logits should be finite
            assert torch.isfinite(logits).all()

            # Each sample should have 4 class predictions
            class_preds = logits.argmax(dim=1)
            assert (class_preds >= 0).all() and (class_preds < 4).all()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
