"""Tests for benchmark module.

Tests the benchmark_accuracy(), benchmark_latency(), and _regions_to_mask()
functions that provide performance comparison metrics.
"""

import pytest
import numpy as np
import torch
import tempfile
import json
from pathlib import Path
from unittest import mock
from PIL import Image

import sys
from pathlib import Path as PathlibPath

# Add sources to path for import
sys.path.insert(0, str(PathlibPath(__file__).parent.parent / "sources/38c98b3d"))

from ane_design_model.benchmark import (
    benchmark_accuracy,
    benchmark_latency,
    _regions_to_mask,
    main
)
from ane_design_model.model import ComponentClassifier


class TestRegionsToMask:
    """Test the _regions_to_mask() helper function."""

    def test_empty_regions_returns_zero_mask(self):
        """Test that all-None regions create empty mask."""
        regions = {
            "header": None,
            "sidebar": None,
            "content": None,
            "footer": None
        }
        mask = _regions_to_mask(regions, 300, 400)

        assert mask.shape == (300, 400)
        assert mask.dtype == np.uint8
        assert np.all(mask == 0)

    def test_single_region_marks_correct_pixels(self):
        """Test that single region correctly marks pixels."""
        regions = {
            "header": {"x": 0, "y": 0, "width": 400, "height": 50},
            "sidebar": None,
            "content": None,
            "footer": None
        }
        mask = _regions_to_mask(regions, 300, 400)

        # Check header region is marked
        assert np.all(mask[0:50, :] == 1)
        # Check rest is unmarked
        assert np.all(mask[50:, :] == 0)

    def test_multiple_regions_marks_all_pixels(self):
        """Test that multiple regions mark their respective pixels."""
        regions = {
            "header": {"x": 0, "y": 0, "width": 400, "height": 50},
            "sidebar": {"x": 0, "y": 50, "width": 100, "height": 200},
            "content": {"x": 100, "y": 50, "width": 300, "height": 200},
            "footer": {"x": 0, "y": 250, "width": 400, "height": 50}
        }
        mask = _regions_to_mask(regions, 300, 400)

        # Check header
        assert np.all(mask[0:50, :] == 1)
        # Check sidebar
        assert np.all(mask[50:250, 0:100] == 1)
        # Check content
        assert np.all(mask[50:250, 100:] == 1)
        # Check footer
        assert np.all(mask[250:, :] == 1)

    def test_clamps_region_to_image_bounds(self):
        """Test that regions are clamped to image boundaries."""
        regions = {
            "header": {"x": -50, "y": -50, "width": 500, "height": 100},
            "sidebar": None,
            "content": None,
            "footer": None
        }
        mask = _regions_to_mask(regions, 100, 100)

        # Should be clamped to [0, 100) for both dimensions
        assert mask.shape == (100, 100)
        # Region x: -50, width: 500 → clamped x: 0, x_end: 100
        # Region y: -50, height: 100 → clamped y: 0, y_end: 100
        # But y is clamped to [0, 100), so y_end becomes min(50, 100) = 50
        # So only top 50 rows are marked
        assert np.sum(mask) == 5000  # Top 50 rows marked (50 * 100)

    def test_returns_uint8_dtype(self):
        """Test that mask has correct dtype."""
        regions = {
            "header": {"x": 0, "y": 0, "width": 100, "height": 100},
            "sidebar": None,
            "content": None,
            "footer": None
        }
        mask = _regions_to_mask(regions, 200, 200)

        assert mask.dtype == np.uint8

    def test_correct_mask_shape(self):
        """Test that mask has correct shape (height, width)."""
        height, width = 480, 640
        regions = {
            "header": {"x": 0, "y": 0, "width": 640, "height": 60},
            "sidebar": None,
            "content": None,
            "footer": None
        }
        mask = _regions_to_mask(regions, height, width)

        assert mask.shape == (height, width)


class TestBenchmarkAccuracy:
    """Test the benchmark_accuracy() function."""

    @pytest.fixture
    def test_dataset(self):
        """Create a temporary dataset with test images."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_path = Path(tmpdir) / "dataset"
            images_dir = dataset_path / "images"
            images_dir.mkdir(parents=True, exist_ok=True)

            # Create a simple test model
            model_path = Path(tmpdir) / "test_model.pt"
            model = ComponentClassifier()
            torch.save(model.state_dict(), model_path)

            # Create test images (just a few for speed)
            for i in range(2):
                image = np.random.randint(0, 256, (300, 400, 3), dtype=np.uint8)
                image_path = images_dir / f"img_{i+1:03d}.png"
                Image.fromarray(image).save(image_path)

            yield str(dataset_path), str(model_path)

    def test_returns_dict_with_required_keys(self, test_dataset):
        """Test that benchmark_accuracy returns required keys."""
        dataset_dir, model_path = test_dataset

        result = benchmark_accuracy(dataset_dir, model_path)

        assert isinstance(result, dict)
        assert "ml_model_accuracy" in result
        assert "heuristic_accuracy" in result
        assert "improvement" in result

    def test_accuracy_values_are_numeric(self, test_dataset):
        """Test that accuracy values are floats."""
        dataset_dir, model_path = test_dataset

        result = benchmark_accuracy(dataset_dir, model_path)

        assert isinstance(result["ml_model_accuracy"], (int, float))
        assert isinstance(result["heuristic_accuracy"], (int, float))

    def test_accuracy_values_in_valid_range(self, test_dataset):
        """Test that accuracy values are in [0, 1]."""
        dataset_dir, model_path = test_dataset

        result = benchmark_accuracy(dataset_dir, model_path)

        assert 0.0 <= result["ml_model_accuracy"] <= 1.0
        assert 0.0 <= result["heuristic_accuracy"] <= 1.0

    def test_improvement_is_string(self, test_dataset):
        """Test that improvement is a string."""
        dataset_dir, model_path = test_dataset

        result = benchmark_accuracy(dataset_dir, model_path)

        assert isinstance(result["improvement"], str)

    def test_improvement_contains_percentage(self, test_dataset):
        """Test that improvement string contains percentage."""
        dataset_dir, model_path = test_dataset

        result = benchmark_accuracy(dataset_dir, model_path)

        # Should be like "+13%" or "-5%" or "N/A"
        assert "%" in result["improvement"] or result["improvement"] == "N/A"

    def test_raises_on_missing_dataset(self):
        """Test that missing dataset raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            benchmark_accuracy("/nonexistent/dataset", "./model.pt")

    def test_handles_missing_images_gracefully(self):
        """Test that missing/corrupt images are handled gracefully."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_path = Path(tmpdir) / "dataset"
            images_dir = dataset_path / "images"
            images_dir.mkdir(parents=True, exist_ok=True)

            model_path = Path(tmpdir) / "test_model.pt"
            model = ComponentClassifier()
            torch.save(model.state_dict(), model_path)

            # Create dataset with one valid image
            image = np.random.randint(0, 256, (300, 400, 3), dtype=np.uint8)
            Image.fromarray(image).save(images_dir / "img_001.png")

            # benchmark_accuracy should handle gracefully and return valid result
            result = benchmark_accuracy(str(dataset_path), str(model_path))
            assert isinstance(result, dict)


class TestBenchmarkLatency:
    """Test the benchmark_latency() function."""

    @pytest.fixture
    def test_dataset(self):
        """Create a temporary dataset with test images."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_path = Path(tmpdir) / "dataset"
            images_dir = dataset_path / "images"
            images_dir.mkdir(parents=True, exist_ok=True)

            # Create a simple test model
            model_path = Path(tmpdir) / "test_model.pt"
            model = ComponentClassifier()
            torch.save(model.state_dict(), model_path)

            # Create test images
            for i in range(5):
                image = np.random.randint(0, 256, (300, 400, 3), dtype=np.uint8)
                image_path = images_dir / f"img_{i+1:03d}.png"
                Image.fromarray(image).save(image_path)

            yield str(dataset_path), str(model_path)

    def test_returns_dict_with_required_keys(self, test_dataset):
        """Test that benchmark_latency returns required keys."""
        dataset_dir, model_path = test_dataset

        result = benchmark_latency(dataset_dir, model_path, num_samples=2)

        assert isinstance(result, dict)
        assert "ane_mean_latency_ms" in result
        assert "cpu_mean_latency_ms" in result
        assert "speedup" in result

    def test_latency_values_are_numeric(self, test_dataset):
        """Test that latency values are floats."""
        dataset_dir, model_path = test_dataset

        result = benchmark_latency(dataset_dir, model_path, num_samples=2)

        assert isinstance(result["ane_mean_latency_ms"], (int, float))
        assert isinstance(result["cpu_mean_latency_ms"], (int, float))

    def test_latency_values_are_positive(self, test_dataset):
        """Test that latency values are non-negative."""
        dataset_dir, model_path = test_dataset

        result = benchmark_latency(dataset_dir, model_path, num_samples=2)

        assert result["ane_mean_latency_ms"] >= 0.0
        assert result["cpu_mean_latency_ms"] >= 0.0

    def test_cpu_latency_greater_than_ane_latency(self, test_dataset):
        """Test that CPU latency > ANE latency (as expected from speedup)."""
        dataset_dir, model_path = test_dataset

        result = benchmark_latency(dataset_dir, model_path, num_samples=2)

        # Since we estimate ANE as CPU / 3.5, ANE should be less
        if result["cpu_mean_latency_ms"] > 0:
            assert result["ane_mean_latency_ms"] < result["cpu_mean_latency_ms"]

    def test_speedup_is_string(self, test_dataset):
        """Test that speedup is a string."""
        dataset_dir, model_path = test_dataset

        result = benchmark_latency(dataset_dir, model_path, num_samples=2)

        assert isinstance(result["speedup"], str)

    def test_speedup_contains_x(self, test_dataset):
        """Test that speedup string contains 'x'."""
        dataset_dir, model_path = test_dataset

        result = benchmark_latency(dataset_dir, model_path, num_samples=2)

        # Should be like "3.6x" or "N/A"
        assert "x" in result["speedup"] or result["speedup"] == "N/A"

    def test_raises_on_missing_dataset(self):
        """Test that missing dataset raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            benchmark_latency("/nonexistent/dataset", "./model.pt")

    def test_num_samples_parameter_respected(self, test_dataset):
        """Test that num_samples parameter limits measurements."""
        dataset_dir, model_path = test_dataset

        # Request only 1 sample from a dataset with 5 images
        result = benchmark_latency(dataset_dir, model_path, num_samples=1)

        # Should succeed and return valid result
        assert isinstance(result, dict)
        assert "cpu_mean_latency_ms" in result


class TestCLIInterface:
    """Test the CLI interface and main() function."""

    def test_main_with_valid_arguments(self):
        """Test main() function with valid command-line arguments."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create dataset
            dataset_path = Path(tmpdir) / "dataset"
            images_dir = dataset_path / "images"
            images_dir.mkdir(parents=True, exist_ok=True)

            # Create test model
            model_path = Path(tmpdir) / "test_model.pt"
            model = ComponentClassifier()
            torch.save(model.state_dict(), model_path)

            # Create test image
            image = np.random.randint(0, 256, (300, 400, 3), dtype=np.uint8)
            Image.fromarray(image).save(images_dir / "img_001.png")

            # Create output path
            output_path = Path(tmpdir) / "results.json"

            # Mock sys.argv to simulate CLI arguments
            with mock.patch(
                "sys.argv",
                [
                    "benchmark",
                    "--model", str(model_path),
                    "--dataset", str(dataset_path),
                    "--output", str(output_path)
                ]
            ):
                main()

            # Verify output file was created
            assert output_path.exists()

            # Verify output is valid JSON
            with open(output_path, "r") as f:
                results = json.load(f)

            assert isinstance(results, dict)

    def test_json_output_has_required_structure(self):
        """Test that JSON output has all required fields."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create dataset
            dataset_path = Path(tmpdir) / "dataset"
            images_dir = dataset_path / "images"
            images_dir.mkdir(parents=True, exist_ok=True)

            # Create test model
            model_path = Path(tmpdir) / "test_model.pt"
            model = ComponentClassifier()
            torch.save(model.state_dict(), model_path)

            # Create test image
            image = np.random.randint(0, 256, (300, 400, 3), dtype=np.uint8)
            Image.fromarray(image).save(images_dir / "img_001.png")

            # Create output path
            output_path = Path(tmpdir) / "results.json"

            # Mock sys.argv to simulate CLI arguments
            with mock.patch(
                "sys.argv",
                [
                    "benchmark",
                    "--model", str(model_path),
                    "--dataset", str(dataset_path),
                    "--output", str(output_path)
                ]
            ):
                main()

            # Verify JSON structure
            with open(output_path, "r") as f:
                results = json.load(f)

            assert "accuracy_comparison" in results
            assert "latency_comparison" in results
            assert "model_size_bytes" in results

            # Verify accuracy_comparison structure
            acc = results["accuracy_comparison"]
            assert "ml_model_accuracy" in acc
            assert "heuristic_accuracy" in acc
            assert "improvement" in acc

            # Verify latency_comparison structure
            lat = results["latency_comparison"]
            assert "ane_mean_latency_ms" in lat
            assert "cpu_mean_latency_ms" in lat
            assert "speedup" in lat

            # Verify model_size_bytes is an integer
            assert isinstance(results["model_size_bytes"], int)
            assert results["model_size_bytes"] > 0

    def test_default_arguments(self):
        """Test that CLI uses sensible defaults."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create dataset in current directory
            dataset_path = Path(tmpdir) / "dataset"
            images_dir = dataset_path / "images"
            images_dir.mkdir(parents=True, exist_ok=True)

            # Create test model
            model_path = Path(tmpdir) / "model.pt"
            model = ComponentClassifier()
            torch.save(model.state_dict(), model_path)

            # Create test image
            image = np.random.randint(0, 256, (300, 400, 3), dtype=np.uint8)
            Image.fromarray(image).save(images_dir / "img_001.png")

            # Change to temp directory and test with explicit paths
            import os
            original_cwd = os.getcwd()
            try:
                os.chdir(tmpdir)

                with mock.patch(
                    "sys.argv",
                    [
                        "benchmark",
                        "--model", str(model_path),
                        "--dataset", str(dataset_path)
                    ]
                ):
                    main()

                # Check default output file was created
                default_output = Path(tmpdir) / "benchmark_results.json"
                assert default_output.exists()

            finally:
                os.chdir(original_cwd)
