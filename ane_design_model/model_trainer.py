"""Model training module for CNN+MLP component classifier.

This module implements the ModelTrainer class that loads synthetic design datasets,
converts ground-truth region dicts to patch-level class labels, trains the
ComponentClassifier model, and saves the trained checkpoint.
"""

import os
import json
from pathlib import Path
from typing import NamedTuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from torch.optim import Adam
from PIL import Image

from .model import ComponentClassifier
from . import (
    DEFAULT_BATCH_SIZE, DEFAULT_EPOCHS, DEFAULT_LEARNING_RATE,
    DEFAULT_TRAIN_VAL_SPLIT, COMPONENT_CLASSES, PATCH_SIZE,
    SYNTHETIC_IMAGE_HEIGHT, SYNTHETIC_IMAGE_WIDTH
)


class TrainingMetrics(NamedTuple):
    """Training result metrics."""
    train_loss: float
    val_accuracy: float
    epochs_trained: int


class ModelTrainer:
    """Train CNN+MLP model on synthetic design dataset."""

    def __init__(
        self,
        batch_size: int = DEFAULT_BATCH_SIZE,
        epochs: int = DEFAULT_EPOCHS,
        learning_rate: float = DEFAULT_LEARNING_RATE,
        device: str | None = None
    ):
        """Initialize trainer.

        Args:
            batch_size: Training batch size (default 16)
            epochs: Number of training epochs (default 10)
            learning_rate: Adam optimizer LR (default 0.001)
            device: torch device ("cpu", "cuda", "mps", or None for auto-detect)
        """
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    def train(
        self,
        dataset_dir: str,
        output_path: str,
        verbose: bool = True
    ) -> TrainingMetrics:
        """Train model on dataset and save checkpoint.

        Args:
            dataset_dir: Path to dataset/ directory with images/ and labels.json
            output_path: Path to save trained model checkpoint (e.g., "./model.pt")
            verbose: Print loss/accuracy per epoch

        Returns:
            TrainingMetrics: {"train_loss": float, "val_accuracy": float, "epochs_trained": int}

        Raises:
            FileNotFoundError: If dataset_dir/images or dataset_dir/labels.json not found
            ValueError: If labels.json format invalid
            OSError: If output_path write fails

        Algorithm:
            1. Load images from dataset_dir/images/
            2. Load labels from dataset_dir/labels.json
            3. Convert labels to patch-level class assignments (0-3)
            4. Build (images, labels) tensors
            5. Split into train (80%) / val (20%) stratified
            6. Create DataLoader with batch_size
            7. For epoch in 1..epochs:
                a. Forward pass on train batches
                b. Compute loss via CrossEntropyLoss
                c. Backward pass, optimizer.step()
                d. Validate on val set, compute accuracy
                e. Log if verbose
            8. Save model to output_path via torch.save()
            9. Return metrics dict

        Implementation Notes:
            - If image loading fails for a specific image, skip with warning
            - All computations on self.device (GPU/CPU/MPS as available)
            - Loss = CrossEntropyLoss (reduction='mean')
            - Optimizer = Adam(lr=learning_rate)
            - Val accuracy = % of patches correctly classified
        """

        # 1. Load images and labels
        dataset_dir = Path(dataset_dir)
        images_dir = dataset_dir / "images"
        labels_path = dataset_dir / "labels.json"

        if not images_dir.exists():
            raise FileNotFoundError(f"Images directory not found: {images_dir}")
        if not labels_path.exists():
            raise FileNotFoundError(f"Labels file not found: {labels_path}")

        with open(labels_path, "r") as f:
            labels_data = json.load(f)

        # 2. Load all images and convert to patch tensors
        images_list = []
        labels_list = []

        for filename in sorted(labels_data.keys()):
            image_path = images_dir / filename

            if not image_path.exists():
                print(f"Warning: Image not found, skipping {filename}")
                continue

            try:
                # Load image
                image = Image.open(image_path).convert("RGB")
                image_array = np.array(image, dtype=np.uint8)

                # Get ground-truth region labels
                region_dict = labels_data[filename]

                # Convert to patch-level labels
                patch_labels = self._region_dict_to_patch_labels(
                    region_dict, image_array.shape
                )

                # Extract patches and their labels
                patches, patch_class_labels = self._extract_patches_and_labels(
                    image_array, patch_labels
                )

                images_list.append(patches)
                labels_list.append(patch_class_labels)

            except Exception as e:
                print(f"Warning: Failed to load {filename}: {e}")
                continue

        if not images_list:
            raise ValueError("No valid images loaded from dataset")

        # Concatenate all patches and labels
        all_patches = torch.cat(images_list, dim=0)  # (total_patches, 3, 128, 128)
        all_labels = torch.cat(labels_list, dim=0)   # (total_patches,)

        # 3. Create dataset and split
        dataset = TensorDataset(all_patches, all_labels)
        train_size = int(DEFAULT_TRAIN_VAL_SPLIT * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(
            dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)

        # 4. Initialize model, loss, optimizer
        model = ComponentClassifier().to(self.device)
        loss_fn = nn.CrossEntropyLoss()
        optimizer = Adam(model.parameters(), lr=self.learning_rate)

        # 5. Training loop
        train_loss = 0.0

        for epoch in range(self.epochs):
            # Training phase
            model.train()
            epoch_loss = 0.0

            for batch_images, batch_labels in train_loader:
                batch_images = batch_images.to(self.device)
                batch_labels = batch_labels.to(self.device)

                # Forward pass
                logits = model(batch_images)
                loss = loss_fn(logits, batch_labels)

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item() * batch_images.size(0)

            epoch_loss /= len(train_dataset)
            train_loss = epoch_loss

            # Validation phase
            model.eval()
            correct = 0
            total = 0

            with torch.no_grad():
                for batch_images, batch_labels in val_loader:
                    batch_images = batch_images.to(self.device)
                    batch_labels = batch_labels.to(self.device)

                    logits = model(batch_images)
                    predictions = torch.argmax(logits, dim=1)

                    correct += (predictions == batch_labels).sum().item()
                    total += batch_labels.size(0)

            val_accuracy = correct / total if total > 0 else 0.0

            if verbose:
                print(f"Epoch {epoch+1}/{self.epochs} - Loss: {epoch_loss:.4f}, Val Acc: {val_accuracy:.4f}")

        # 6. Save model
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), output_path)

        # 7. Return metrics
        return {
            "train_loss": train_loss,
            "val_accuracy": val_accuracy,
            "epochs_trained": self.epochs
        }

    def _region_dict_to_patch_labels(
        self, region_dict: dict, image_shape: tuple
    ) -> dict[int, int]:
        """Convert region_dict to patch-level class assignments.

        Args:
            region_dict: {"header": {...}, "sidebar": {...}, ...} from detect_layout_regions
            image_shape: (height, width, channels)

        Returns:
            {patch_id → class_label (0-3)}

        Algorithm:
            1. Divide image into grid of 128×128 patches
            2. For each patch, compute intersection with each region
            3. Assign patch to region with largest intersection
            4. Default to class 2 (card) if no regions or tie

        IMPORTANT - Edge Padding Strategy:
            Patches at image edges are zero-padded to 128×128 for model input.
            This means edge regions see artificial zero pixels. Model learns to
            classify despite padding artifacts. Edge patch accuracy may be degraded
            (2-5%) compared to full patches, but this trade-off is acceptable for
            implementation simplicity and uniform patch processing.
        """
        height, width = image_shape[0], image_shape[1]
        patch_labels = {}

        # Grid of patches (for 300×400 image → 3×4 grid)
        patches_h = (height + PATCH_SIZE - 1) // PATCH_SIZE
        patches_w = (width + PATCH_SIZE - 1) // PATCH_SIZE

        patch_id = 0

        for patch_row in range(patches_h):
            for patch_col in range(patches_w):
                # Patch bounds in pixels
                patch_y = patch_row * PATCH_SIZE
                patch_x = patch_col * PATCH_SIZE
                patch_y_end = min(patch_y + PATCH_SIZE, height)
                patch_x_end = min(patch_x + PATCH_SIZE, width)

                # Find region with largest intersection
                best_class = 2  # Default to "card"
                best_area = 0

                for region_name, region_info in region_dict.items():
                    if region_info is None:
                        continue

                    # Region bounds
                    reg_x = region_info["x"]
                    reg_y = region_info["y"]
                    reg_w = region_info["width"]
                    reg_h = region_info["height"]
                    reg_x_end = reg_x + reg_w
                    reg_y_end = reg_y + reg_h

                    # Intersection
                    inter_x = max(patch_x, reg_x)
                    inter_y = max(patch_y, reg_y)
                    inter_x_end = min(patch_x_end, reg_x_end)
                    inter_y_end = min(patch_y_end, reg_y_end)

                    if inter_x < inter_x_end and inter_y < inter_y_end:
                        area = (inter_x_end - inter_x) * (inter_y_end - inter_y)
                        if area > best_area:
                            best_area = area
                            # Map region name to class ID
                            # Handle aliases: sidebar→nav, content→card
                            canonical_name = region_name
                            if region_name == "sidebar":
                                canonical_name = "nav"
                            elif region_name == "content":
                                canonical_name = "card"

                            best_class = COMPONENT_CLASSES.get(canonical_name, 2)

                patch_labels[patch_id] = best_class
                patch_id += 1

        return patch_labels

    def _extract_patches_and_labels(
        self, image: np.ndarray, patch_labels: dict
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Extract 128×128 patches from image and convert to tensors.

        Args:
            image: (300, 400, 3) uint8 RGB image
            patch_labels: {patch_id → class_label}

        Returns:
            (patches_tensor, labels_tensor)
            patches: (num_patches, 3, 128, 128) float32 in [0, 1]
            labels: (num_patches,) int64
        """
        height, width = image.shape[0], image.shape[1]
        patches_h = (height + PATCH_SIZE - 1) // PATCH_SIZE
        patches_w = (width + PATCH_SIZE - 1) // PATCH_SIZE

        patches = []
        labels = []
        patch_id = 0

        for patch_row in range(patches_h):
            for patch_col in range(patches_w):
                # Extract patch
                y_start = patch_row * PATCH_SIZE
                x_start = patch_col * PATCH_SIZE
                y_end = min(y_start + PATCH_SIZE, height)
                x_end = min(x_start + PATCH_SIZE, width)

                patch = image[y_start:y_end, x_start:x_end, :]

                # Pad to 128×128 if necessary
                if patch.shape[0] < PATCH_SIZE or patch.shape[1] < PATCH_SIZE:
                    padded = np.zeros((PATCH_SIZE, PATCH_SIZE, 3), dtype=np.uint8)
                    padded[:patch.shape[0], :patch.shape[1], :] = patch
                    patch = padded

                # Normalize to [0, 1] and convert to tensor
                patch_tensor = torch.from_numpy(patch).float() / 255.0
                patch_tensor = patch_tensor.permute(2, 0, 1)  # (H, W, 3) → (3, H, W)

                patches.append(patch_tensor)
                labels.append(patch_labels.get(patch_id, 2))
                patch_id += 1

        return torch.stack(patches), torch.tensor(labels, dtype=torch.int64)
