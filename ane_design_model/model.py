"""CNN+MLP architecture for patch-based component classification.

This module defines the ComponentClassifier model for mapping 128×128 image patches
to component types (header, nav, card, footer).
"""

import torch
import torch.nn as nn


class ComponentClassifier(nn.Module):
    """CNN+MLP for patch-based component classification.

    Input: (batch_size, 3, 128, 128) float32 in [0, 1]
    Output: (batch_size, 4) float32 logits (no softmax; CrossEntropyLoss handles it)
    """

    def __init__(self):
        """Initialize model with default PyTorch weight initialization."""
        super().__init__()

        # CNN backbone
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, stride=2)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, stride=2)

        # Flatten computed output size: 64 * 32 * 32 = 65536
        self.fc1 = nn.Linear(65536, 32)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(32, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: (batch_size, 3, 128, 128) float32 tensor

        Returns:
            (batch_size, 4) float32 logits

        Raises:
            RuntimeError: If input shape invalid
        """
        # Validate input shape
        if x.ndim != 4 or x.shape[1] != 3 or x.shape[2] != 128 or x.shape[3] != 128:
            raise RuntimeError(
                f"Expected input shape (batch, 3, 128, 128), got {x.shape}"
            )

        # Conv block 1
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)  # → (batch, 32, 64, 64)

        # Conv block 2
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)  # → (batch, 64, 32, 32)

        # Flatten and MLP
        x = x.flatten(1)   # → (batch, 65536)
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)    # → (batch, 4)

        return x


def create_model() -> ComponentClassifier:
    """Create and return a ComponentClassifier instance."""
    return ComponentClassifier()
