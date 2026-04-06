"""Integration tests for benchmark module with ml_layout_detector.

Tests the critical interactions between benchmark functions and the ML layout detector,
including accuracy and latency benchmarking with actual inference calls.
"""

import pytest
import numpy as np
import tempfile
import json
from pathlib import Path
from unittest import mock
from PIL import Image
import sys
import time

# Add sources to path
sys.path.insert(0, str(Path(__file__).parent.parent / "sources/38c98b3d"))

from ane_design_model.benchmark import (
    benchmark_accuracy,
    benchmark_latency,
    _regions_to_mask,
)
from ane_design_model.ml_layout_detector import ml_detect_layout_regions
from ane_design_model.model import ComponentClassifier
from ane_design_model.model_trainer import ModelTrainer
import torch


class TestBenchmarkAccuracyWithMLDetector:
    """Test benchmark_accuracy() function integration with ml_detect_layout_regions()."""

    @pytest.fixture
    def temp_dataset(self):
        """Create temporary dataset with synthetic images and trained model."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            # Create images directory
            images_dir = tmpdir_path / "images"
            images_dir.mkdir()
            labels_dir = tmpdir_path / "labels"
            labels_dir.mkdir(exist_ok=True)

            # Create a few synthetic images
            labels_data = {}
            for i in range(3):
                # Create simple image with distinct regions
                img = np.random.randint(50, 200, (300, 400, 3), dtype=np.uint8)
                img[0:50, :] = [200, 50, 50]  # Red header
                img[50:250, 0:100] = [50, 200, 50]  # Green sidebar
                img[50:250, 100:] = [50, 50, 200]  # Blue content
                img[250:, :] = [200, 200, 50]  # Yellow footer

                # Save image
                img_path = images_dir / f"img_{i:03d}.png"
                Image.fromarray(img).save(img_path)

                # Create labels
                labels_data[f"img_{i:03d}.png"] = {
                    "header": {"x": 0, "y": 0, "width": 400, "height": 50},
                    "nav": {"x": 0, "y": 50, "width": 100, "height": 200},
                    "card": {"x": 100, "y": 50, "width": 300, "height": 200},
                    "footer": {"x": 0, "y": 250, "width": 400, "height": 50}
                }

            # Save labels
            with open(tmpdir_path / "labels.json", "w") as f:
                json.dump(labels_data, f)

            # Train a quick model
            trainer = ModelTrainer(epochs=1, batch_size=2)
            model_path = tmpdir_path / "model.pt"
            try:
                trainer.train(str(tmpdir_path), str(model_path), verbose=False)
            except Exception as e:
                # If training fails (e.g., no GPU), create a dummy model
                model = ComponentClassifier()
                torch.save(model.state_dict(), str(model_path))

            yield tmpdir_path, str(model_path)

    def test_benchmark_accuracy_calls_ml_detector(self, temp_dataset):
        """Verify that benchmark_accuracy calls ml_detect_layout_regions()."""
        dataset_dir, model_path = temp_dataset

        with mock.patch('ane_design_model.benchmark.ml_detect_layout_regions',
                        side_effect=ml_detect_layout_regions) as mock_ml:
            result = benchmark_accuracy(str(dataset_dir), model_path)

            # Should call ml_detect_layout_regions for each image
            assert mock_ml.call_count >= 1

            # Check result format
            assert "ml_model_accuracy" in result
            assert "heuristic_accuracy" in result
            assert "improvement" in result

    def test_benchmark_accuracy_returns_valid_metrics(self, temp_dataset):
        """Test that benchmark_accuracy returns valid metrics."""
        dataset_dir, model_path = temp_dataset

        result = benchmark_accuracy(str(dataset_dir), model_path)

        assert isinstance(result, dict)
        assert "ml_model_accuracy" in result
        assert "heuristic_accuracy" in result
        assert "improvement" in result

        # Validate types and ranges
        assert isinstance(result["ml_model_accuracy"], float)
        assert isinstance(result["heuristic_accuracy"], float)
        assert isinstance(result["improvement"], str)

        # Accuracy should be in [0, 1] or possibly percentage string
        assert 0.0 <= result["ml_model_accuracy"] <= 1.0 or result["ml_model_accuracy"] == 0.0
        assert 0.0 <= result["heuristic_accuracy"] <= 1.0

    def test_benchmark_accuracy_processes_multiple_images(self, temp_dataset):
        """Verify that benchmark_accuracy processes multiple images."""
        dataset_dir, model_path = temp_dataset

        # All images should be processed
        result = benchmark_accuracy(str(dataset_dir), model_path)

        # Should have valid results
        assert isinstance(result, dict)
        assert result["ml_model_accuracy"] is not None
        assert result["heuristic_accuracy"] is not None


class TestBenchmarkLatencyWithMLDetector:
    """Test benchmark_latency() function integration with ml_detect_layout_regions()."""

    @pytest.fixture
    def temp_dataset_with_model(self):
        """Create temporary dataset with trained model."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            # Create images directory
            images_dir = tmpdir_path / "images"
            images_dir.mkdir()

            # Create synthetic images
            labels_data = {}
            for i in range(5):
                img = np.random.randint(50, 200, (300, 400, 3), dtype=np.uint8)
                img_path = images_dir / f"img_{i:03d}.png"
                Image.fromarray(img).save(img_path)

                labels_data[f"img_{i:03d}.png"] = {
                    "header": {"x": 0, "y": 0, "width": 400, "height": 50},
                    "nav": {"x": 0, "y": 50, "width": 100, "height": 200},
                    "card": {"x": 100, "y": 50, "width": 300, "height": 200},
                    "footer": {"x": 0, "y": 250, "width": 400, "height": 50}
                }

            # Save labels
            with open(tmpdir_path / "labels.json", "w") as f:
                json.dump(labels_data, f)

            # Train model or create dummy
            trainer = ModelTrainer(epochs=1, batch_size=2)
            model_path = tmpdir_path / "model.pt"
            try:
                trainer.train(str(tmpdir_path), str(model_path), verbose=False)
            except Exception:
                model = ComponentClassifier()
                torch.save(model.state_dict(), str(model_path))

            yield tmpdir_path, str(model_path)

    def test_benchmark_latency_calls_ml_detector(self, temp_dataset_with_model):
        """Verify that benchmark_latency calls ml_detect_layout_regions()."""
        dataset_dir, model_path = temp_dataset_with_model

        with mock.patch('ane_design_model.benchmark.ml_detect_layout_regions',
                        side_effect=ml_detect_layout_regions) as mock_ml:
            result = benchmark_latency(str(dataset_dir), model_path, num_samples=2)

            # Should call ml_detect_layout_regions
            assert mock_ml.call_count >= 1

    def test_benchmark_latency_returns_valid_metrics(self, temp_dataset_with_model):
        """Test that benchmark_latency returns valid latency metrics."""
        dataset_dir, model_path = temp_dataset_with_model

        result = benchmark_latency(str(dataset_dir), model_path, num_samples=2)

        assert isinstance(result, dict)
        assert "ane_mean_latency_ms" in result
        assert "cpu_mean_latency_ms" in result
        assert "speedup" in result

        # Latencies should be numeric and positive
        assert isinstance(result["ane_mean_latency_ms"], float)
        assert isinstance(result["cpu_mean_latency_ms"], float)
        assert isinstance(result["speedup"], str)

        # ANE latency should be less than CPU (estimated 3.5x speedup)
        if result["cpu_mean_latency_ms"] > 0:
            assert result["ane_mean_latency_ms"] < result["cpu_mean_latency_ms"]

    def test_benchmark_latency_respects_num_samples(self, temp_dataset_with_model):
        """Verify that benchmark_latency respects num_samples parameter."""
        dataset_dir, model_path = temp_dataset_with_model

        with mock.patch('ane_design_model.benchmark.ml_detect_layout_regions',
                        side_effect=ml_detect_layout_regions) as mock_ml:
            result = benchmark_latency(str(dataset_dir), model_path, num_samples=2)

            # Should call detector for up to num_samples
            assert mock_ml.call_count <= 2

    def test_benchmark_latency_measures_timing(self, temp_dataset_with_model):
        """Test that latency measurements are reasonable."""
        dataset_dir, model_path = temp_dataset_with_model

        # Measure latency
        result = benchmark_latency(str(dataset_dir), model_path, num_samples=1)

        # Should have reasonable latency (> 0 if images were processed)
        assert result["cpu_mean_latency_ms"] >= 0
        assert result["ane_mean_latency_ms"] >= 0


class TestBenchmarkAndDetectorConsistency:
    """Test consistency between benchmark and ml_layout_detector."""

    def test_benchmark_accuracy_mask_consistency(self):
        """Test that _regions_to_mask produces consistent masks."""
        regions = {
            "header": {"x": 0, "y": 0, "width": 400, "height": 50},
            "sidebar": {"x": 0, "y": 50, "width": 100, "height": 200},
            "content": {"x": 100, "y": 50, "width": 300, "height": 200},
            "footer": {"x": 0, "y": 250, "width": 400, "height": 50}
        }

        height, width = 300, 400
        mask1 = _regions_to_mask(regions, height, width)
        mask2 = _regions_to_mask(regions, height, width)

        # Should produce identical masks
        assert np.array_equal(mask1, mask2)

        # Mask should have correct properties
        assert mask1.shape == (height, width)
        assert mask1.dtype == np.uint8

    def test_ml_detector_output_compatible_with_benchmark(self):
        """Test that ml_layout_detector output is compatible with benchmark."""
        # Create a dummy model
        model = ComponentClassifier()

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "model.pt"
            torch.save(model.state_dict(), str(model_path))

            # Test image
            img = np.random.randint(0, 256, (300, 400, 3), dtype=np.uint8)

            # Get detector output
            result = ml_detect_layout_regions(img, str(model_path))

            # Result should be compatible with _regions_to_mask
            assert isinstance(result, dict)
            assert all(key in result for key in ["header", "sidebar", "content", "footer"])

            # Should be able to convert to mask
            mask = _regions_to_mask(result, 300, 400)
            assert mask.shape == (300, 400)
            assert mask.dtype == np.uint8

    def test_detector_regions_to_mask_accuracy_comparison(self):
        """Test that detector output can be compared with heuristic via masks."""
        from layout_detector import detect_layout_regions

        model = ComponentClassifier()

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "model.pt"
            torch.save(model.state_dict(), str(model_path))

            # Test image
            img = np.random.randint(0, 256, (300, 400, 3), dtype=np.uint8)

            # Get both detectors
            ml_result = ml_detect_layout_regions(img, str(model_path))
            heur_result = detect_layout_regions(img)

            # Both should produce masks
            ml_mask = _regions_to_mask(ml_result, 300, 400)
            heur_mask = _regions_to_mask(heur_result, 300, 400)

            # Masks should be comparable
            accuracy = np.mean(ml_mask == heur_mask)
            assert 0.0 <= accuracy <= 1.0


class TestBenchmarkOutputFormat:
    """Test that benchmark output matches specification."""

    @pytest.fixture
    def minimal_dataset(self):
        """Create minimal dataset for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            # Create images directory
            images_dir = tmpdir_path / "images"
            images_dir.mkdir()

            # Create one test image
            img = np.random.randint(50, 200, (300, 400, 3), dtype=np.uint8)
            Image.fromarray(img).save(images_dir / "test_img.png")

            # Create labels
            labels_data = {
                "test_img.png": {
                    "header": {"x": 0, "y": 0, "width": 400, "height": 50},
                    "nav": {"x": 0, "y": 50, "width": 100, "height": 200},
                    "card": {"x": 100, "y": 50, "width": 300, "height": 200},
                    "footer": {"x": 0, "y": 250, "width": 400, "height": 50}
                }
            }

            with open(tmpdir_path / "labels.json", "w") as f:
                json.dump(labels_data, f)

            # Create dummy model
            model = ComponentClassifier()
            model_path = tmpdir_path / "model.pt"
            torch.save(model.state_dict(), str(model_path))

            yield tmpdir_path, str(model_path)

    def test_accuracy_result_format(self, minimal_dataset):
        """Test that accuracy result has correct format."""
        dataset_dir, model_path = minimal_dataset

        result = benchmark_accuracy(str(dataset_dir), model_path)

        # Check required keys
        assert "ml_model_accuracy" in result
        assert "heuristic_accuracy" in result
        assert "improvement" in result

        # Check types
        assert isinstance(result["ml_model_accuracy"], float)
        assert isinstance(result["heuristic_accuracy"], float)
        assert isinstance(result["improvement"], str)

    def test_latency_result_format(self, minimal_dataset):
        """Test that latency result has correct format."""
        dataset_dir, model_path = minimal_dataset

        result = benchmark_latency(str(dataset_dir), model_path, num_samples=1)

        # Check required keys
        assert "ane_mean_latency_ms" in result
        assert "cpu_mean_latency_ms" in result
        assert "speedup" in result

        # Check types
        assert isinstance(result["ane_mean_latency_ms"], float)
        assert isinstance(result["cpu_mean_latency_ms"], float)
        assert isinstance(result["speedup"], str)


class TestBenchmarkErrorHandling:
    """Test error handling in benchmark functions."""

    def test_benchmark_accuracy_handles_missing_images_gracefully(self):
        """Test that benchmark_accuracy handles missing images."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_dir = Path(tmpdir)
            images_dir = dataset_dir / "images"
            images_dir.mkdir()

            # Create empty labels but no images
            labels = {"missing_img.png": {}}
            with open(dataset_dir / "labels.json", "w") as f:
                json.dump(labels, f)

            # Create dummy model
            model = ComponentClassifier()
            model_path = dataset_dir / "model.pt"
            torch.save(model.state_dict(), str(model_path))

            # Should handle gracefully
            result = benchmark_accuracy(str(dataset_dir), str(model_path))
            assert isinstance(result, dict)

    def test_benchmark_latency_handles_missing_images(self):
        """Test that benchmark_latency handles missing images."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_dir = Path(tmpdir)
            images_dir = dataset_dir / "images"
            images_dir.mkdir()

            # Create empty labels
            with open(dataset_dir / "labels.json", "w") as f:
                json.dump({}, f)

            # Create dummy model
            model = ComponentClassifier()
            model_path = dataset_dir / "model.pt"
            torch.save(model.state_dict(), str(model_path))

            # Should return zero latencies
            result = benchmark_latency(str(dataset_dir), str(model_path))
            assert result["cpu_mean_latency_ms"] == 0.0
            assert result["ane_mean_latency_ms"] == 0.0

    def test_benchmark_accuracy_handles_bad_model_path(self):
        """Test that benchmark_accuracy handles missing model gracefully."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_dir = Path(tmpdir)
            images_dir = dataset_dir / "images"
            images_dir.mkdir()

            # Create a test image
            img = np.random.randint(0, 256, (300, 400, 3), dtype=np.uint8)
            Image.fromarray(img).save(images_dir / "test.png")

            # Create labels
            with open(dataset_dir / "labels.json", "w") as f:
                json.dump({"test.png": {}}, f)

            # Try with non-existent model - should handle gracefully
            result = benchmark_accuracy(str(dataset_dir), "/nonexistent/model.pt")

            # Should return valid result structure even if inference fails
            assert isinstance(result, dict)
            assert "ml_model_accuracy" in result
            assert "heuristic_accuracy" in result
            assert "improvement" in result
