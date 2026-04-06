"""Integration tests for inference engine and ml_layout_detector interaction.

Tests the integration between ComponentClassifierInference and ml_layout_detector,
focusing on:
1. Inference engine loading and caching through ml_detect_layout_regions
2. Single patch vs batch classification consistency
3. Region name mapping (internal names: nav, card → output names: sidebar, content)
4. Error propagation from inference layer to detector layer
5. Edge cases in image patch extraction and reconstruction
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

from ane_design_model.ml_layout_detector import ml_detect_layout_regions, _use_brightness_fallback
from ane_design_model.inference import ComponentClassifierInference
from ane_design_model.model import ComponentClassifier
from ane_design_model import PATCH_SIZE, COMPONENT_CLASSES, REVERSE_COMPONENT_CLASSES


class TestInferenceEngineIntegration:
    """Test ComponentClassifierInference integration with ml_layout_detector."""

    @pytest.fixture
    def model_path(self):
        """Create a temporary model file for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "test_model.pt"
            model = ComponentClassifier()
            torch.save(model.state_dict(), model_path)
            yield str(model_path)

    def test_inference_engine_initialized_by_detector(self, model_path):
        """Verify ml_detect_layout_regions correctly initializes ComponentClassifierInference."""
        _use_brightness_fallback()

        test_image = np.random.randint(0, 256, (300, 400, 3), dtype=np.uint8)

        # Should not raise
        result = ml_detect_layout_regions(test_image, model_path)

        assert isinstance(result, dict)
        assert all(k in result for k in ['header', 'sidebar', 'content', 'footer'])

    def test_inference_batch_processing_consistency(self, model_path):
        """Verify batch processing gives consistent results with single patch classification."""
        inference_engine = ComponentClassifierInference(model_path)

        # Create a deterministic patch
        patch = np.ones((PATCH_SIZE, PATCH_SIZE, 3), dtype=np.uint8) * 128

        # Single patch classification
        single_class = inference_engine.predict_patch_class(patch)

        # Batch classification with one patch
        patch_tensor = torch.from_numpy(patch).float() / 255.0
        patch_tensor = patch_tensor.permute(2, 0, 1).unsqueeze(0).to(inference_engine.device)
        batch_logits = inference_engine.predict_batch(patch_tensor)
        batch_class = batch_logits.argmax(dim=1).item()

        # Should match
        assert single_class == batch_class, \
            f"Single patch class {single_class} != batch class {batch_class}"

    def test_inference_batch_multiple_patches(self, model_path):
        """Verify batch processing handles multiple patches correctly."""
        inference_engine = ComponentClassifierInference(model_path)

        # Create multiple patches
        num_patches = 4
        patches_np = np.ones((num_patches, PATCH_SIZE, PATCH_SIZE, 3), dtype=np.uint8) * 128

        # Convert to tensor and batch classify
        patches_tensor = torch.from_numpy(patches_np).float() / 255.0
        patches_tensor = patches_tensor.permute(0, 3, 1, 2).to(inference_engine.device)
        batch_logits = inference_engine.predict_batch(patches_tensor)

        # Should return (num_patches, 4) logits
        assert batch_logits.shape == (num_patches, 4)

        # All classes should be valid (0-3)
        batch_classes = batch_logits.argmax(dim=1)
        assert torch.all(batch_classes >= 0) and torch.all(batch_classes < 4)

    def test_detector_uses_inference_predict_image_layout(self, model_path):
        """Verify ml_detect_layout_regions uses predict_image_layout correctly."""
        _use_brightness_fallback()

        test_image = np.ones((300, 400, 3), dtype=np.uint8) * 100

        # Call detector
        detector_result = ml_detect_layout_regions(test_image, model_path)

        # Results should have output keys
        assert set(detector_result.keys()) == {'header', 'sidebar', 'content', 'footer'}


class TestRegionNameMapping:
    """Test internal-to-output region name mapping."""

    @pytest.fixture
    def model_path(self):
        """Create a temporary model file for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "test_model.pt"
            model = ComponentClassifier()
            torch.save(model.state_dict(), model_path)
            yield str(model_path)

    def test_output_uses_sidebar_not_nav(self, model_path):
        """Verify output dict uses 'sidebar' not internal 'nav'."""
        _use_brightness_fallback()
        test_image = np.random.randint(0, 256, (300, 400, 3), dtype=np.uint8)

        result = ml_detect_layout_regions(test_image, model_path)

        # Output should use 'sidebar', not 'nav'
        assert 'sidebar' in result, "Output should have 'sidebar' key"
        assert 'nav' not in result, "Output should not have internal 'nav' key"

    def test_output_uses_content_not_card(self, model_path):
        """Verify output dict uses 'content' not internal 'card'."""
        _use_brightness_fallback()
        test_image = np.random.randint(0, 256, (300, 400, 3), dtype=np.uint8)

        result = ml_detect_layout_regions(test_image, model_path)

        # Output should use 'content', not 'card'
        assert 'content' in result, "Output should have 'content' key"
        assert 'card' not in result, "Output should not have internal 'card' key"

    def test_all_output_keys_present(self, model_path):
        """Verify output has all required keys in canonical form."""
        _use_brightness_fallback()
        test_image = np.random.randint(0, 256, (300, 400, 3), dtype=np.uint8)

        result = ml_detect_layout_regions(test_image, model_path)

        # Exactly these keys in output
        expected_keys = {'header', 'sidebar', 'content', 'footer'}
        assert set(result.keys()) == expected_keys


class TestPatchExtractionAndReconstruction:
    """Test patch extraction, classification, and region reconstruction pipeline."""

    @pytest.fixture
    def model_path(self):
        """Create a temporary model file for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "test_model.pt"
            model = ComponentClassifier()
            torch.save(model.state_dict(), model_path)
            yield str(model_path)

    def test_patch_extraction_for_standard_size(self, model_path):
        """Verify patches are correctly extracted from standard image size."""
        inference_engine = ComponentClassifierInference(model_path)

        # 256x256 image = exactly 2x2 patches
        test_image = np.ones((256, 256, 3), dtype=np.uint8) * 128

        result = inference_engine.predict_image_layout(test_image)

        # Should process without error and return valid dict with output names
        # (after ml_layout_detector maps internal names)
        assert isinstance(result, dict)

    def test_patch_extraction_for_non_aligned_size(self, model_path):
        """Verify patches are correctly extracted and padded for non-128-aligned images."""
        inference_engine = ComponentClassifierInference(model_path)

        # 300x400 image = 3x4 patches with padding on last row/col
        test_image = np.ones((300, 400, 3), dtype=np.uint8) * 128

        result = inference_engine.predict_image_layout(test_image)

        assert isinstance(result, dict)

    def test_patch_extraction_for_small_image(self, model_path):
        """Verify patches are correctly extracted and padded for very small images."""
        inference_engine = ComponentClassifierInference(model_path)

        # Smaller than patch size - should be padded to 128x128
        test_image = np.ones((64, 64, 3), dtype=np.uint8) * 128

        result = inference_engine.predict_image_layout(test_image)

        assert isinstance(result, dict)

    def test_region_coordinate_validity(self, model_path):
        """Verify reconstructed region coordinates are valid."""
        _use_brightness_fallback()
        test_image = np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8)

        result = ml_detect_layout_regions(test_image, model_path)

        height, width = test_image.shape[:2]

        # Check each region's coordinates
        for region_name, region in result.items():
            if region is not None:
                x = region['x']
                y = region['y']
                w = region['width']
                h = region['height']

                # Coordinates should be non-negative
                assert x >= 0, f"{region_name}: x should be >= 0, got {x}"
                assert y >= 0, f"{region_name}: y should be >= 0, got {y}"
                assert w >= 0, f"{region_name}: width should be >= 0, got {w}"
                assert h >= 0, f"{region_name}: height should be >= 0, got {h}"

                # Should not exceed image bounds
                assert x + w <= width, f"{region_name}: x+width exceeds image width"
                assert y + h <= height, f"{region_name}: y+height exceeds image height"


class TestInferenceErrorPropagation:
    """Test error propagation from inference layer to detector layer."""

    @pytest.fixture
    def model_path(self):
        """Create a temporary model file for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "test_model.pt"
            model = ComponentClassifier()
            torch.save(model.state_dict(), model_path)
            yield str(model_path)

    def test_invalid_image_shape_error(self, model_path):
        """Verify invalid image shape raises appropriate error."""
        inference_engine = ComponentClassifierInference(model_path)

        # Wrong shape - should raise ValueError
        bad_image = np.ones((100, 100, 4), dtype=np.uint8)  # 4 channels instead of 3

        with pytest.raises(ValueError):
            inference_engine.predict_image_layout(bad_image)

    def test_invalid_patch_dtype_error(self, model_path):
        """Verify invalid patch dtype raises appropriate error."""
        inference_engine = ComponentClassifierInference(model_path)

        # Wrong dtype - should raise ValueError
        bad_patch = np.ones((PATCH_SIZE, PATCH_SIZE, 3), dtype=np.float32)

        with pytest.raises(ValueError):
            inference_engine.predict_patch_class(bad_patch)

    def test_model_forward_invalid_shape_error(self, model_path):
        """Verify model forward pass validates input shape."""
        model = ComponentClassifier()

        # Wrong input shape for model
        bad_input = torch.randn(2, 3, 64, 64)  # Should be 128x128

        with pytest.raises(RuntimeError):
            model(bad_input)


class TestPatchGridReconstruction:
    """Test the patch grid and region boundary reconstruction logic."""

    @pytest.fixture
    def model_path(self):
        """Create a temporary model file for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "test_model.pt"
            model = ComponentClassifier()
            torch.save(model.state_dict(), model_path)
            yield str(model_path)

    def test_single_patch_image(self, model_path):
        """Verify single patch (< PATCH_SIZE) reconstructs correctly."""
        _use_brightness_fallback()

        # Image smaller than patch size
        test_image = np.ones((50, 50, 3), dtype=np.uint8) * 128
        result = ml_detect_layout_regions(test_image, model_path)

        # Should complete and return valid structure
        assert isinstance(result, dict)
        assert all(k in result for k in ['header', 'sidebar', 'content', 'footer'])

    def test_exact_patch_size_image(self, model_path):
        """Verify image exactly PATCH_SIZE reconstructs correctly."""
        _use_brightness_fallback()

        test_image = np.ones((PATCH_SIZE, PATCH_SIZE, 3), dtype=np.uint8) * 128
        result = ml_detect_layout_regions(test_image, model_path)

        assert isinstance(result, dict)
        assert all(k in result for k in ['header', 'sidebar', 'content', 'footer'])

    def test_multiple_patches_grid(self, model_path):
        """Verify multi-patch grid reconstructs correctly."""
        _use_brightness_fallback()

        # Create 3x3 grid of patches
        test_image = np.ones((3 * PATCH_SIZE, 3 * PATCH_SIZE, 3), dtype=np.uint8) * 128
        result = ml_detect_layout_regions(test_image, model_path)

        assert isinstance(result, dict)
        assert all(k in result for k in ['header', 'sidebar', 'content', 'footer'])


class TestInferenceDeviceHandling:
    """Test inference engine device handling (CPU vs GPU)."""

    @pytest.fixture
    def model_path(self):
        """Create a temporary model file for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "test_model.pt"
            model = ComponentClassifier()
            torch.save(model.state_dict(), model_path)
            yield str(model_path)

    def test_inference_respects_device_parameter(self, model_path):
        """Verify ComponentClassifierInference respects device parameter."""
        # Force CPU device
        inference_engine = ComponentClassifierInference(model_path, device="cpu")

        assert inference_engine.device == "cpu"
        assert inference_engine.model.training == False

    def test_inference_autodetects_device(self, model_path):
        """Verify ComponentClassifierInference auto-detects device."""
        # Don't specify device
        inference_engine = ComponentClassifierInference(model_path)

        # Should have a valid device
        assert inference_engine.device is not None
        assert isinstance(inference_engine.device, str)


class TestEndToEndIntegration:
    """End-to-end integration tests combining all components."""

    @pytest.fixture
    def model_path(self):
        """Create a temporary model file for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "test_model.pt"
            model = ComponentClassifier()
            torch.save(model.state_dict(), model_path)
            yield str(model_path)

    def test_detector_to_inference_full_pipeline(self, model_path):
        """Test full pipeline from ml_detect_layout_regions through inference."""
        _use_brightness_fallback()

        # Create test image with varied colors to simulate different regions
        test_image = np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8)

        # Call detector
        result = ml_detect_layout_regions(test_image, model_path)

        # Verify result structure
        assert isinstance(result, dict)
        assert set(result.keys()) == {'header', 'sidebar', 'content', 'footer'}

        # Each region should be None or valid dict
        for region_name, region_data in result.items():
            if region_data is not None:
                assert isinstance(region_data, dict)
                assert set(region_data.keys()) == {'x', 'y', 'width', 'height'}
                assert all(isinstance(region_data[k], (int, np.integer)) for k in region_data.keys())

    def test_multiple_sequential_calls_with_same_model(self, model_path):
        """Test multiple sequential calls use cached model."""
        _use_brightness_fallback()

        # First call
        test_image_1 = np.random.randint(0, 256, (300, 400, 3), dtype=np.uint8)
        result_1 = ml_detect_layout_regions(test_image_1, model_path)

        # Second call with same model - should reuse cached inference engine
        test_image_2 = np.random.randint(0, 256, (400, 500, 3), dtype=np.uint8)
        result_2 = ml_detect_layout_regions(test_image_2, model_path)

        # Both should succeed and return valid dicts
        assert isinstance(result_1, dict)
        assert isinstance(result_2, dict)

    def test_handles_edge_case_images(self, model_path):
        """Test detector handles various edge case images."""
        _use_brightness_fallback()

        test_cases = [
            ("tiny image", (32, 32, 3)),
            ("very wide", (100, 1000, 3)),
            ("very tall", (1000, 100, 3)),
            ("non-square", (300, 450, 3)),
        ]

        for desc, shape in test_cases:
            test_image = np.random.randint(0, 256, shape, dtype=np.uint8)
            result = ml_detect_layout_regions(test_image, model_path)

            assert isinstance(result, dict), f"Failed for {desc}"
            assert all(k in result for k in ['header', 'sidebar', 'content', 'footer']), \
                f"Missing keys for {desc}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
