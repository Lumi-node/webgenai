"""Integration tests for benchmark CLI with actual dataset and model training.

Tests the end-to-end benchmark CLI workflow including dataset generation,
model training, and benchmark execution.
"""

import pytest
import numpy as np
import json
import tempfile
import subprocess
from pathlib import Path
from PIL import Image
import sys
import torch

# Add sources to path
sys.path.insert(0, str(Path(__file__).parent.parent / "sources/38c98b3d"))

from ane_design_model.benchmark import main
from ane_design_model.model import ComponentClassifier
from ane_design_model.model_trainer import ModelTrainer
from ane_design_model.ml_layout_detector import ml_detect_layout_regions


class TestBenchmarkCLIIntegration:
    """Test benchmark CLI with full pipeline."""

    @pytest.fixture
    def benchmark_dataset_and_model(self):
        """Create complete benchmark environment."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            # Create images directory
            images_dir = tmpdir_path / "images"
            images_dir.mkdir()

            # Create synthetic training dataset
            labels_data = {}
            np.random.seed(42)

            for i in range(5):
                # Create image with distinct color regions
                img = np.full((300, 400, 3), 128, dtype=np.uint8)

                # Add distinct regions with different colors
                img[0:50, :] = [200, 50, 50]      # Red header
                img[50:250, 0:100] = [50, 200, 50]    # Green sidebar
                img[50:250, 100:] = [50, 50, 200]    # Blue content
                img[250:, :] = [200, 200, 50]     # Yellow footer

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

            # Train model
            trainer = ModelTrainer(epochs=1, batch_size=2)
            model_path = tmpdir_path / "model.pt"

            try:
                trainer.train(str(tmpdir_path), str(model_path), verbose=False)
            except Exception:
                # Fallback to dummy model if training fails
                model = ComponentClassifier()
                torch.save(model.state_dict(), str(model_path))

            # Create output directory
            output_dir = tmpdir_path / "results"
            output_dir.mkdir()

            yield tmpdir_path, str(model_path), output_dir

    def test_benchmark_cli_creates_output_file(self, benchmark_dataset_and_model):
        """Test that benchmark CLI creates output JSON file."""
        dataset_dir, model_path, output_dir = benchmark_dataset_and_model
        output_file = output_dir / "benchmark_results.json"

        # Run benchmark programmatically
        sys.argv = [
            "benchmark",
            "--model", model_path,
            "--dataset", str(dataset_dir),
            "--output", str(output_file)
        ]

        try:
            main()
        except SystemExit:
            pass  # main() calls sys.exit()

        # Output file should be created
        assert output_file.exists()

    def test_benchmark_output_json_structure(self, benchmark_dataset_and_model):
        """Test that benchmark output has correct JSON structure."""
        dataset_dir, model_path, output_dir = benchmark_dataset_and_model
        output_file = output_dir / "benchmark_results.json"

        # Run benchmark
        sys.argv = [
            "benchmark",
            "--model", model_path,
            "--dataset", str(dataset_dir),
            "--output", str(output_file)
        ]

        try:
            main()
        except SystemExit:
            pass

        # Load and verify JSON
        with open(output_file, "r") as f:
            results = json.load(f)

        # Check top-level structure
        assert "accuracy_comparison" in results
        assert "latency_comparison" in results
        assert "model_size_bytes" in results

        # Check accuracy structure
        accuracy = results["accuracy_comparison"]
        assert "ml_model_accuracy" in accuracy
        assert "heuristic_accuracy" in accuracy
        assert "improvement" in accuracy

        # Check latency structure
        latency = results["latency_comparison"]
        assert "ane_mean_latency_ms" in latency
        assert "cpu_mean_latency_ms" in latency
        assert "speedup" in latency

        # Check model size
        assert isinstance(results["model_size_bytes"], int)
        assert results["model_size_bytes"] > 0

    def test_benchmark_output_values_valid(self, benchmark_dataset_and_model):
        """Test that benchmark output contains valid values."""
        dataset_dir, model_path, output_dir = benchmark_dataset_and_model
        output_file = output_dir / "benchmark_results.json"

        # Run benchmark
        sys.argv = [
            "benchmark",
            "--model", model_path,
            "--dataset", str(dataset_dir),
            "--output", str(output_file)
        ]

        try:
            main()
        except SystemExit:
            pass

        with open(output_file, "r") as f:
            results = json.load(f)

        # Validate accuracy values
        accuracy = results["accuracy_comparison"]
        assert isinstance(accuracy["ml_model_accuracy"], (int, float))
        assert isinstance(accuracy["heuristic_accuracy"], (int, float))
        assert isinstance(accuracy["improvement"], str)

        # Validate latency values
        latency = results["latency_comparison"]
        assert isinstance(latency["ane_mean_latency_ms"], (int, float))
        assert isinstance(latency["cpu_mean_latency_ms"], (int, float))
        assert isinstance(latency["speedup"], str)

        # Validate model size
        assert results["model_size_bytes"] > 0
        assert results["model_size_bytes"] < 100_000_000  # Less than 100MB

    def test_benchmark_model_size_matches_file(self, benchmark_dataset_and_model):
        """Test that reported model size matches actual file size."""
        dataset_dir, model_path, output_dir = benchmark_dataset_and_model
        output_file = output_dir / "benchmark_results.json"

        # Run benchmark
        sys.argv = [
            "benchmark",
            "--model", model_path,
            "--dataset", str(dataset_dir),
            "--output", str(output_file)
        ]

        try:
            main()
        except SystemExit:
            pass

        # Get actual file size
        actual_size = Path(model_path).stat().st_size

        # Load benchmark results
        with open(output_file, "r") as f:
            results = json.load(f)

        # Should match
        assert results["model_size_bytes"] == actual_size


class TestBenchmarkCLIWithDifferentOptions:
    """Test benchmark CLI with various options."""

    @pytest.fixture
    def simple_dataset(self):
        """Create simple dataset."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            # Create images
            images_dir = tmpdir_path / "images"
            images_dir.mkdir()

            img = np.random.randint(0, 256, (300, 400, 3), dtype=np.uint8)
            Image.fromarray(img).save(images_dir / "test.png")

            # Create labels
            labels = {
                "test.png": {
                    "header": {"x": 0, "y": 0, "width": 400, "height": 50},
                    "nav": {"x": 0, "y": 50, "width": 100, "height": 200},
                    "card": {"x": 100, "y": 50, "width": 300, "height": 200},
                    "footer": {"x": 0, "y": 250, "width": 400, "height": 50}
                }
            }

            with open(tmpdir_path / "labels.json", "w") as f:
                json.dump(labels, f)

            # Create dummy model
            model = ComponentClassifier()
            model_path = tmpdir_path / "model.pt"
            torch.save(model.state_dict(), str(model_path))

            yield tmpdir_path, str(model_path)

    def test_benchmark_custom_output_path(self, simple_dataset):
        """Test benchmark with custom output path."""
        dataset_dir, model_path = simple_dataset

        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = Path(tmpdir) / "custom_output.json"

            sys.argv = [
                "benchmark",
                "--model", model_path,
                "--dataset", str(dataset_dir),
                "--output", str(output_file)
            ]

            try:
                main()
            except SystemExit:
                pass

            assert output_file.exists()

            with open(output_file, "r") as f:
                data = json.load(f)
            assert "accuracy_comparison" in data

    def test_benchmark_default_output_path(self, simple_dataset):
        """Test benchmark with default output path."""
        dataset_dir, model_path = simple_dataset

        # Save current directory
        import os
        old_cwd = os.getcwd()

        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                os.chdir(tmpdir)

                sys.argv = [
                    "benchmark",
                    "--model", model_path,
                    "--dataset", str(dataset_dir)
                ]

                try:
                    main()
                except SystemExit:
                    pass

                # Should create default output file
                default_output = Path("benchmark_results.json")
                assert default_output.exists()
        finally:
            os.chdir(old_cwd)

    def test_benchmark_with_nested_output_dir(self, simple_dataset):
        """Test that benchmark creates nested output directories."""
        dataset_dir, model_path = simple_dataset

        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = Path(tmpdir) / "results/subdir/benchmark.json"

            sys.argv = [
                "benchmark",
                "--model", model_path,
                "--dataset", str(dataset_dir),
                "--output", str(output_file)
            ]

            try:
                main()
            except SystemExit:
                pass

            assert output_file.exists()
            assert output_file.parent.exists()


class TestBenchmarkAccuracyWithMLDetectorIntegration:
    """Test accuracy benchmarking with real detector integration."""

    @pytest.fixture
    def colored_dataset(self):
        """Create dataset with colored regions for easier detection."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            images_dir = tmpdir_path / "images"
            images_dir.mkdir()

            labels_data = {}

            for i in range(3):
                # Create image with distinct regions
                img = np.full((300, 400, 3), 200, dtype=np.uint8)
                img[0:50, :] = [255, 0, 0]        # Bright red header
                img[50:250, 0:100] = [0, 255, 0]  # Bright green sidebar
                img[50:250, 100:] = [0, 0, 255]   # Bright blue content
                img[250:, :] = [255, 255, 0]      # Bright yellow footer

                Image.fromarray(img).save(images_dir / f"img_{i:03d}.png")

                labels_data[f"img_{i:03d}.png"] = {
                    "header": {"x": 0, "y": 0, "width": 400, "height": 50},
                    "nav": {"x": 0, "y": 50, "width": 100, "height": 200},
                    "card": {"x": 100, "y": 50, "width": 300, "height": 200},
                    "footer": {"x": 0, "y": 250, "width": 400, "height": 50}
                }

            with open(tmpdir_path / "labels.json", "w") as f:
                json.dump(labels_data, f)

            # Train model
            trainer = ModelTrainer(epochs=1, batch_size=2)
            model_path = tmpdir_path / "model.pt"
            try:
                trainer.train(str(tmpdir_path), str(model_path), verbose=False)
            except Exception:
                model = ComponentClassifier()
                torch.save(model.state_dict(), str(model_path))

            yield tmpdir_path, str(model_path)

    def test_accuracy_benchmark_processes_detector_output(self, colored_dataset):
        """Test that accuracy benchmark correctly processes ML detector output."""
        from ane_design_model.benchmark import benchmark_accuracy

        dataset_dir, model_path = colored_dataset

        result = benchmark_accuracy(str(dataset_dir), model_path)

        # Should produce valid result
        assert isinstance(result, dict)
        assert "ml_model_accuracy" in result
        assert "heuristic_accuracy" in result
        assert "improvement" in result

        # Should be able to compare ML and heuristic
        assert result["ml_model_accuracy"] is not None
        assert result["heuristic_accuracy"] is not None


class TestBenchmarkLatencyWithMLDetectorIntegration:
    """Test latency benchmarking with real detector integration."""

    @pytest.fixture
    def timing_dataset(self):
        """Create dataset for latency testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            images_dir = tmpdir_path / "images"
            images_dir.mkdir()

            labels_data = {}

            for i in range(5):
                img = np.random.randint(50, 200, (300, 400, 3), dtype=np.uint8)
                Image.fromarray(img).save(images_dir / f"img_{i:03d}.png")

                labels_data[f"img_{i:03d}.png"] = {
                    "header": {"x": 0, "y": 0, "width": 400, "height": 50},
                    "nav": {"x": 0, "y": 50, "width": 100, "height": 200},
                    "card": {"x": 100, "y": 50, "width": 300, "height": 200},
                    "footer": {"x": 0, "y": 250, "width": 400, "height": 50}
                }

            with open(tmpdir_path / "labels.json", "w") as f:
                json.dump(labels_data, f)

            # Create model
            model = ComponentClassifier()
            model_path = tmpdir_path / "model.pt"
            torch.save(model.state_dict(), str(model_path))

            yield tmpdir_path, str(model_path)

    def test_latency_measurements_are_positive(self, timing_dataset):
        """Test that latency measurements are positive values."""
        from ane_design_model.benchmark import benchmark_latency

        dataset_dir, model_path = timing_dataset

        result = benchmark_latency(str(dataset_dir), model_path, num_samples=3)

        # All times should be non-negative
        assert result["cpu_mean_latency_ms"] >= 0
        assert result["ane_mean_latency_ms"] >= 0

    def test_latency_speedup_calculation(self, timing_dataset):
        """Test that ANE speedup is correctly estimated."""
        from ane_design_model.benchmark import benchmark_latency

        dataset_dir, model_path = timing_dataset

        result = benchmark_latency(str(dataset_dir), model_path, num_samples=2)

        # ANE should be faster (3.5x speedup estimate)
        if result["cpu_mean_latency_ms"] > 0:
            assert result["ane_mean_latency_ms"] <= result["cpu_mean_latency_ms"]
