"""Tests for ml_layout_detector module.

Tests the ml_detect_layout_regions() function which provides a drop-in replacement
for the brightness-based layout_detector.detect_layout_regions().
"""

import pytest
import numpy as np
import torch
import tempfile
from pathlib import Path
from unittest import mock

import sys
from pathlib import Path as PathlibPath

# Add sources to path for import
sys.path.insert(0, str(PathlibPath(__file__).parent.parent / "sources/38c98b3d"))

from ane_design_model.ml_layout_detector import (
    ml_detect_layout_regions,
    _use_brightness_fallback
)
from ane_design_model.inference import ComponentClassifierInference
from ane_design_model.model import ComponentClassifier

# Import LayoutDetectionError through importlib to handle the numeric module name
import importlib.util
types_spec = importlib.util.spec_from_file_location(
    "project_types_test",
    PathlibPath(__file__).parent.parent / "sources/38c98b3d/types.py"
)
project_types_test = importlib.util.module_from_spec(types_spec)
types_spec.loader.exec_module(project_types_test)
LayoutDetectionError = project_types_test.LayoutDetectionError


class TestMLDetectLayoutRegionsFunctionSignature:
    """Test function signature matches detect_layout_regions()."""

    def test_function_signature_parameters(self):
        """Verify function accepts correct parameters."""
        import inspect
        sig = inspect.signature(ml_detect_layout_regions)
        params = list(sig.parameters.keys())

        # Should have 'image' and 'model_path' parameters
        assert 'image' in params, "Missing 'image' parameter"
        assert 'model_path' in params, "Missing 'model_path' parameter"

        # model_path should have default value
        assert sig.parameters['model_path'].default == "./model.pt"

    def test_function_return_type_annotation(self):
        """Verify function has return type annotation."""
        import inspect
        sig = inspect.signature(ml_detect_layout_regions)
        # Return annotation should be 'dict'
        assert sig.return_annotation == dict or sig.return_annotation == "dict"


class TestMLDetectLayoutRegionsReturnType:
    """Test return type is correct dict structure."""

    @pytest.fixture
    def mock_model_path(self):
        """Create a temporary model file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "test_model.pt"
            # Create and save a dummy model
            model = ComponentClassifier()
            torch.save(model.state_dict(), model_path)
            yield str(model_path)

    def test_returns_dict_with_correct_keys(self, mock_model_path):
        """Verify return dict has required keys."""
        # Create a small test image
        test_image = np.random.randint(0, 256, (300, 400, 3), dtype=np.uint8)

        result = ml_detect_layout_regions(test_image, mock_model_path)

        assert isinstance(result, dict), "Result should be a dict"

        # Should have all four required keys
        required_keys = {'header', 'sidebar', 'content', 'footer'}
        assert set(result.keys()) == required_keys, f"Expected keys {required_keys}, got {set(result.keys())}"

    def test_each_region_is_none_or_region_dict(self, mock_model_path):
        """Verify each region is either None or a valid region dict."""
        test_image = np.random.randint(0, 256, (300, 400, 3), dtype=np.uint8)

        result = ml_detect_layout_regions(test_image, mock_model_path)

        for region_name, region_value in result.items():
            assert region_value is None or isinstance(region_value, dict), \
                f"{region_name} should be None or dict, got {type(region_value)}"

            if region_value is not None:
                # If it's a dict, it should have these keys
                expected_keys = {'x', 'y', 'width', 'height'}
                assert set(region_value.keys()) == expected_keys, \
                    f"{region_name} dict should have keys {expected_keys}, got {set(region_value.keys())}"

                # All values should be integers
                for key in expected_keys:
                    assert isinstance(region_value[key], (int, np.integer)), \
                        f"{region_name}[{key}] should be int, got {type(region_value[key])}"


class TestSingletonCaching:
    """Test module-level singleton caching behavior."""

    @pytest.fixture
    def mock_model_path(self):
        """Create a temporary model file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "test_model.pt"
            model = ComponentClassifier()
            torch.save(model.state_dict(), model_path)
            yield str(model_path)

    def test_cache_hit_same_model_path(self, mock_model_path):
        """Verify model is not reloaded when called with same path."""
        test_image = np.random.randint(0, 256, (300, 400, 3), dtype=np.uint8)

        # Reset cache first
        _use_brightness_fallback()

        # Mock the ComponentClassifierInference constructor
        with mock.patch('ane_design_model.ml_layout_detector.ComponentClassifierInference') as mock_inf:
            # Create mock instance
            mock_instance = mock.Mock()
            mock_instance.predict_image_layout.return_value = {
                "header": None, "sidebar": None, "content": None, "footer": None
            }
            mock_inf.return_value = mock_instance

            # First call - should create inference engine
            ml_detect_layout_regions(test_image, mock_model_path)
            assert mock_inf.call_count == 1

            # Second call with same path - should reuse cached engine
            ml_detect_layout_regions(test_image, mock_model_path)
            assert mock_inf.call_count == 1, "Constructor should only be called once (cache hit)"

    def test_cache_miss_different_model_path(self, mock_model_path):
        """Verify model is reloaded when called with different path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path_2 = Path(tmpdir) / "test_model_2.pt"
            model = ComponentClassifier()
            torch.save(model.state_dict(), model_path_2)

            test_image = np.random.randint(0, 256, (300, 400, 3), dtype=np.uint8)

            # Reset cache first
            _use_brightness_fallback()

            # Mock the ComponentClassifierInference constructor
            with mock.patch('ane_design_model.ml_layout_detector.ComponentClassifierInference') as mock_inf:
                # Create mock instances
                mock_instance_1 = mock.Mock()
                mock_instance_1.predict_image_layout.return_value = {
                    "header": None, "sidebar": None, "content": None, "footer": None
                }
                mock_instance_2 = mock.Mock()
                mock_instance_2.predict_image_layout.return_value = {
                    "header": None, "sidebar": None, "content": None, "footer": None
                }
                mock_inf.side_effect = [mock_instance_1, mock_instance_2]

                # First call with path 1
                ml_detect_layout_regions(test_image, mock_model_path)
                assert mock_inf.call_count == 1

                # Second call with different path - should create new inference engine
                ml_detect_layout_regions(test_image, str(model_path_2))
                assert mock_inf.call_count == 2, "Constructor should be called for different path (cache miss)"

    def test_reset_cache_with_fallback_function(self, mock_model_path):
        """Verify _use_brightness_fallback() resets the cache."""
        test_image = np.random.randint(0, 256, (300, 400, 3), dtype=np.uint8)

        # Reset cache first
        _use_brightness_fallback()

        with mock.patch('ane_design_model.ml_layout_detector.ComponentClassifierInference') as mock_inf:
            mock_instance = mock.Mock()
            mock_instance.predict_image_layout.return_value = {
                "header": None, "sidebar": None, "content": None, "footer": None
            }
            mock_inf.return_value = mock_instance

            # First call
            ml_detect_layout_regions(test_image, mock_model_path)
            assert mock_inf.call_count == 1

            # Reset cache
            _use_brightness_fallback()

            # Next call should recreate the engine
            ml_detect_layout_regions(test_image, mock_model_path)
            assert mock_inf.call_count == 2, "Cache should be cleared after reset"


class TestErrorHandling:
    """Test error handling behavior."""

    def test_missing_model_file_raises_layout_detection_error(self):
        """Verify missing model file raises LayoutDetectionError (not FileNotFoundError)."""
        test_image = np.random.randint(0, 256, (300, 400, 3), dtype=np.uint8)
        nonexistent_path = "/nonexistent/path/to/model.pt"

        # Reset cache
        _use_brightness_fallback()

        # The LayoutDetectionError is from project_types in the ml_layout_detector module
        # We need to catch it as a base exception type
        with pytest.raises(Exception) as exc_info:
            ml_detect_layout_regions(test_image, nonexistent_path)

        # Should be a LayoutDetectionError
        assert "LayoutDetectionError" in type(exc_info.value).__name__
        # Should mention model not found
        assert "not found" in str(exc_info.value).lower() or "model" in str(exc_info.value).lower()

    def test_inference_error_raises_layout_detection_error(self):
        """Verify inference errors are wrapped in LayoutDetectionError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "test_model.pt"
            model = ComponentClassifier()
            torch.save(model.state_dict(), model_path)

            test_image = np.random.randint(0, 256, (300, 400, 3), dtype=np.uint8)

            # Reset cache
            _use_brightness_fallback()

            # Mock the ComponentClassifierInference to raise an error during predict_image_layout
            with mock.patch('ane_design_model.ml_layout_detector.ComponentClassifierInference') as mock_inf:
                mock_instance = mock.Mock()
                mock_instance.predict_image_layout.side_effect = RuntimeError("Test inference error")
                mock_inf.return_value = mock_instance

                with pytest.raises(Exception) as exc_info:
                    ml_detect_layout_regions(test_image, str(model_path))

                assert "LayoutDetectionError" in type(exc_info.value).__name__
                assert "failed" in str(exc_info.value).lower()

    def test_error_preserves_context(self):
        """Verify error context is preserved with 'from' clause."""
        test_image = np.random.randint(0, 256, (300, 400, 3), dtype=np.uint8)
        nonexistent_path = "/nonexistent/path/to/model.pt"

        # Reset cache
        _use_brightness_fallback()

        with pytest.raises(Exception) as exc_info:
            ml_detect_layout_regions(test_image, nonexistent_path)

        # Check that the exception has a __cause__
        assert exc_info.value.__cause__ is not None, "Exception should preserve context with 'from' clause"


class TestDropInReplacementCompatibility:
    """Test that ml_detect_layout_regions() is compatible as drop-in replacement."""

    @pytest.fixture
    def mock_model_path(self):
        """Create a temporary model file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "test_model.pt"
            model = ComponentClassifier()
            torch.save(model.state_dict(), model_path)
            yield str(model_path)

    def test_compatible_with_detect_layout_regions_output_format(self, mock_model_path):
        """Verify output format matches detect_layout_regions()."""
        test_image = np.random.randint(0, 256, (300, 400, 3), dtype=np.uint8)

        ml_result = ml_detect_layout_regions(test_image, mock_model_path)

        # Output should be a dict with the same structure
        assert isinstance(ml_result, dict)

        # All required keys should be present
        for key in ['header', 'sidebar', 'content', 'footer']:
            assert key in ml_result, f"Missing required key: {key}"

        # Each value should be None or a region dict
        for region in ml_result.values():
            if region is not None:
                assert isinstance(region, dict)
                assert all(k in region for k in ['x', 'y', 'width', 'height'])

    def test_accepts_uint8_rgb_images(self, mock_model_path):
        """Verify function accepts uint8 RGB images like detect_layout_regions()."""
        # Test various image sizes
        for h, w in [(100, 150), (300, 400), (600, 800)]:
            test_image = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)
            result = ml_detect_layout_regions(test_image, mock_model_path)
            assert isinstance(result, dict)

    def test_handles_various_image_sizes(self, mock_model_path):
        """Verify function handles various image dimensions."""
        test_cases = [
            (100, 100),      # Small image
            (300, 400),      # Standard size
            (600, 800),      # Larger image
            (256, 256),      # Power of 2
        ]

        for h, w in test_cases:
            test_image = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)
            result = ml_detect_layout_regions(test_image, mock_model_path)
            assert isinstance(result, dict), f"Failed for image size {h}x{w}"


class TestIntegration:
    """Integration tests with actual model and inference."""

    @pytest.fixture
    def mock_model_path(self):
        """Create a temporary model file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "test_model.pt"
            model = ComponentClassifier()
            torch.save(model.state_dict(), model_path)
            yield str(model_path)

    def test_end_to_end_inference(self, mock_model_path):
        """Test end-to-end inference with real model."""
        # Create a deterministic test image
        test_image = np.ones((300, 400, 3), dtype=np.uint8) * 128

        # Should complete without error
        result = ml_detect_layout_regions(test_image, mock_model_path)

        # Verify result structure
        assert isinstance(result, dict)
        assert all(k in result for k in ['header', 'sidebar', 'content', 'footer'])

    def test_accepts_different_image_dtypes_and_ranges(self, mock_model_path):
        """Verify function handles expected input data types."""
        # Should work with uint8 [0, 255]
        test_image = np.random.randint(0, 256, (300, 400, 3), dtype=np.uint8)
        result = ml_detect_layout_regions(test_image, mock_model_path)
        assert isinstance(result, dict)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
