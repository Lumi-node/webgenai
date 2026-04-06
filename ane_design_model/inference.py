"""Model inference engine with optional ANE acceleration.

This module provides the ComponentClassifierInference class for performing
component classification on design images using a trained CNN+MLP model.
"""

import sys
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import Optional

from .model import ComponentClassifier
from . import (
    PATCH_SIZE, COMPONENT_CLASSES, REVERSE_COMPONENT_CLASSES,
    MODEL_INPUT_CHANNELS, MODEL_INPUT_SIZE, MODEL_NUM_CLASSES
)

# Attempt ANE import (optional, for future integration)
try:
    from ane_trainer import ane_forward_pass
    HAS_ANE = True
except ImportError:
    HAS_ANE = False


class ComponentClassifierInference:
    """Load trained model and perform inference with optional ANE acceleration."""

    def __init__(self, model_path: str, device: str | None = None):
        """Initialize inference engine.

        Args:
            model_path: Path to trained model checkpoint (.pt file)
            device: torch device (None = auto-detect)

        Raises:
            FileNotFoundError: If model_path doesn't exist
            RuntimeError: If model loading fails
        """
        self.model_path = Path(model_path)

        if not self.model_path.exists():
            raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Load model
        self.model = ComponentClassifier().to(self.device)
        try:
            state_dict = torch.load(self.model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}")

        self.model.eval()

    def predict_patch_class(self, patch: np.ndarray) -> int:
        """Classify single patch.

        Args:
            patch: (128, 128, 3) uint8 RGB image

        Returns:
            Class ID (0-3): header (0), nav (1), card (2), footer (3)

        Raises:
            ValueError: If patch shape invalid
            RuntimeError: If model forward pass fails
        """
        # Validate shape
        if patch.shape != (PATCH_SIZE, PATCH_SIZE, 3) or patch.dtype != np.uint8:
            raise ValueError(
                f"Expected ({PATCH_SIZE}, {PATCH_SIZE}, 3) uint8, got {patch.shape} {patch.dtype}"
            )

        # Normalize to [0, 1] and convert to tensor
        patch_tensor = torch.from_numpy(patch).float() / 255.0
        patch_tensor = patch_tensor.permute(2, 0, 1)  # (H, W, 3) → (3, H, W)
        patch_tensor = patch_tensor.unsqueeze(0).to(self.device)  # (1, 3, H, W)

        # Forward pass
        with torch.no_grad():
            logits = self.model(patch_tensor)

        # Get class
        class_id = logits.argmax(dim=1).item()

        return class_id

    def predict_batch(self, patches: torch.Tensor) -> torch.Tensor:
        """Classify batch of patches with optional ANE acceleration.

        Args:
            patches: (batch, 3, 128, 128) float32 tensor on self.device in [0, 1]

        Returns:
            (batch, 4) float32 logits

        Raises:
            RuntimeError: If model forward pass fails

        Implementation Notes:
            - Uses CPU PyTorch forward pass by default
            - If ANE available and Apple Silicon detected: attempts ane_forward_pass()
            - On ANE failure: falls back to CPU automatically
            - All tensor-to-numpy conversions handle dtype/layout correctly
        """
        if HAS_ANE and self._is_apple_silicon():
            try:
                # Convert tensor to numpy for ANE: (B, 3, H, W) → (B, H, W, 3)
                patches_np = patches.permute(0, 2, 3, 1).cpu().numpy()

                # Call ANE forward pass
                logits_np = ane_forward_pass(patches_np, self.model)

                # Convert back to tensor
                logits = torch.from_numpy(logits_np).to(self.device)
                return logits
            except Exception as e:
                # Fallback to CPU on ANE error
                print(f"Warning: ANE inference failed, falling back to CPU: {e}")
                return self._cpu_forward(patches)
        else:
            return self._cpu_forward(patches)

    def _cpu_forward(self, patches: torch.Tensor) -> torch.Tensor:
        """Standard PyTorch forward pass on CPU/GPU/MPS.

        Args:
            patches: (batch, 3, 128, 128) float32 tensor on self.device

        Returns:
            (batch, 4) float32 logits
        """
        with torch.no_grad():
            return self.model(patches)

    def _is_apple_silicon(self) -> bool:
        """Check if running on Apple Silicon (ARM-based Mac).

        Returns:
            True if platform is macOS with ARM processor, False otherwise
        """
        import platform
        return (
            sys.platform == 'darwin' and
            platform.processor() in ('arm64', 'arm')
        )

    def predict_image_layout(self, image: np.ndarray) -> dict:
        """Classify image patches and reconstruct region layout.

        Args:
            image: (H, W, 3) uint8 RGB image

        Returns:
            ComponentDict: {"header": {...}, "sidebar": {...}, "content": {...}, "footer": {...}}
            Each region is None or {"x": int, "y": int, "width": int, "height": int}

        Raises:
            ValueError: If image shape invalid
            RuntimeError: If model forward pass fails

        Algorithm:
            1. Extract all 128×128 patches from image (stride=128, with zero-padding)
            2. Batch classify patches using predict_batch() (CPU or ANE path)
            3. Reconstruct region boundaries via bounding box on patch grid
            4. Convert from internal class names to output names (nav→sidebar, card→content)
            5. Return as ComponentDict

        Implementation Notes:
            - Patches extracted on fixed grid; stride=128
            - If image not multiple of 128, last patch is zero-padded to 128×128
            - Model classifies each patch independently
            - Boundary reconstruction: find min/max patch indices per class → pixel coords
        """

        # Validate shape
        if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
            raise ValueError(
                f"Expected (H, W, 3) uint8, got {image.shape} {image.dtype}"
            )

        height, width = image.shape[0], image.shape[1]

        # 1. Extract patches
        patches = []
        patch_coords = []  # (patch_row, patch_col) for each patch

        for patch_row in range(0, height, PATCH_SIZE):
            for patch_col in range(0, width, PATCH_SIZE):
                # Extract patch region from image
                patch_y_end = min(patch_row + PATCH_SIZE, height)
                patch_x_end = min(patch_col + PATCH_SIZE, width)

                patch = image[patch_row:patch_y_end, patch_col:patch_x_end, :]

                # Zero-pad to PATCH_SIZE if necessary
                if patch.shape[0] < PATCH_SIZE or patch.shape[1] < PATCH_SIZE:
                    padded = np.zeros((PATCH_SIZE, PATCH_SIZE, 3), dtype=np.uint8)
                    padded[:patch.shape[0], :patch.shape[1], :] = patch
                    patch = padded

                patches.append(patch)
                # Store grid indices (not pixel coordinates)
                patch_coords.append((patch_row // PATCH_SIZE, patch_col // PATCH_SIZE))

        if not patches:
            # No patches extracted (empty image)
            return {"header": None, "sidebar": None, "content": None, "footer": None}

        # 2. Batch classify patches using predict_batch()
        patch_batch = np.stack(patches)  # (num_patches, 128, 128, 3) uint8
        patch_batch = torch.from_numpy(patch_batch).float() / 255.0
        patch_batch = patch_batch.permute(0, 3, 1, 2).to(self.device)  # (num, 3, 128, 128)

        logits = self.predict_batch(patch_batch)  # (num_patches, 4)
        class_ids = logits.argmax(dim=1).cpu().numpy()  # (num_patches,)

        # 3. Create patch grid
        max_patch_row = max(coord[0] for coord in patch_coords)
        max_patch_col = max(coord[1] for coord in patch_coords)
        grid_height = max_patch_row + 1
        grid_width = max_patch_col + 1

        patch_grid = np.full((grid_height, grid_width), -1, dtype=np.int32)
        for (patch_row, patch_col), class_id in zip(patch_coords, class_ids):
            patch_grid[patch_row, patch_col] = class_id

        # 4. Reconstruct regions for each class
        regions = {}

        for class_id, region_name in REVERSE_COMPONENT_CLASSES.items():
            # Find all patches of this class
            mask = (patch_grid == class_id)

            if not mask.any():
                # No patches of this class
                regions[region_name] = None
            else:
                # Compute bounding box from patch indices
                patch_rows, patch_cols = np.where(mask)

                min_patch_row = patch_rows.min()
                max_patch_row = patch_rows.max()
                min_patch_col = patch_cols.min()
                max_patch_col = patch_cols.max()

                # Convert patch grid coordinates to pixel coordinates
                x = min_patch_col * PATCH_SIZE
                y = min_patch_row * PATCH_SIZE
                width_px = (max_patch_col - min_patch_col + 1) * PATCH_SIZE
                height_px = (max_patch_row - min_patch_row + 1) * PATCH_SIZE

                # Clamp to image bounds
                x = max(0, x)
                y = max(0, y)
                width_px = min(width_px, width - x)
                height_px = min(height_px, height - y)

                regions[region_name] = {
                    "x": int(x),
                    "y": int(y),
                    "width": int(width_px),
                    "height": int(height_px)
                }

        # 5. Map internal names to output names (nav→sidebar, card→content)
        result = {
            "header": regions.get("header"),
            "sidebar": regions.get("nav"),
            "content": regions.get("card"),
            "footer": regions.get("footer")
        }

        return result
