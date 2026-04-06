"""Synthetic design dataset generator with ground-truth component labels.

Generates synthetic design mockup images paired with ground-truth component annotations
derived from brightness-based layout detection.
"""

import argparse
import json
import sys
import importlib.util
from pathlib import Path
import numpy as np
from PIL import Image

from . import (
    SYNTHETIC_IMAGE_HEIGHT,
    SYNTHETIC_IMAGE_WIDTH,
    SYNTHETIC_IMAGE_CHANNELS,
    LabelsDict,
)

# Load layout_detector module by file path to handle numeric module name
_layout_detector_path = Path(__file__).parent.parent / "sources/38c98b3d/layout_detector.py"
_spec = importlib.util.spec_from_file_location("layout_detector_module", str(_layout_detector_path))
_layout_detector_module = importlib.util.module_from_spec(_spec)
sys.modules['layout_detector_module'] = _layout_detector_module
_spec.loader.exec_module(_layout_detector_module)

detect_layout_regions = _layout_detector_module.detect_layout_regions


class DatasetGenerator:
    """Generate synthetic design mockups with ground-truth component labels."""

    def __init__(self, seed: int = 42):
        """Initialize generator with RNG seed.

        Args:
            seed: Random seed for reproducibility
        """
        self.rng = np.random.RandomState(seed)

    def generate_synthetic_image(self, index: int) -> np.ndarray:
        """Generate single synthetic design mockup.

        Args:
            index: Image index (0-49) for seeded variation

        Returns:
            (300, 400, 3) uint8 RGB image with distinct regions:
            - Header: top 15%, brightness offset +50% or -50% from bg
            - Footer: bottom 15%, brightness offset +50% or -50% from bg
            - Sidebar: left 25%, brightness offset +50% or -50% from bg
            - Content: central region, slightly different brightness
            - Random background color

        Algorithm:
            1. Choose random background color (60-180 intensity to avoid extremes)
            2. Choose random offset (+50% or -50% brightness for regions)
            3. Draw header (y: 0-45)
            4. Draw sidebar (x: 0-100)
            5. Draw footer (y: 255-300)
            6. Draw content (remaining area)
            7. Add subtle noise to each region (±5% intensity variation)
            8. Return as uint8 clipped to [0, 255]
        """
        height, width = SYNTHETIC_IMAGE_HEIGHT, SYNTHETIC_IMAGE_WIDTH

        # Create image array
        image = np.zeros((height, width, SYNTHETIC_IMAGE_CHANNELS), dtype=np.float32)

        # Random background color (60-180 to avoid extremes)
        bg_color = self.rng.uniform(60, 180, 3)
        image[:, :, :] = bg_color

        # Random brightness offset: ±50% of background
        offset_factor = 0.5 * (1 if self.rng.rand() > 0.5 else -1)
        region_color = bg_color + (bg_color * offset_factor)
        region_color = np.clip(region_color, 0, 255)

        # Header region (top 15%, y: 0-45)
        header_height = int(0.15 * height)
        image[0:header_height, :, :] = region_color

        # Footer region (bottom 15%, y: 255-300)
        footer_y = int(0.85 * height)
        image[footer_y:, :, :] = region_color + self.rng.uniform(-10, 10, 3)

        # Sidebar region (left 25%, x: 0-100)
        sidebar_width = int(0.25 * width)
        sidebar_color = bg_color + (bg_color * -offset_factor)
        sidebar_color = np.clip(sidebar_color, 0, 255)
        image[header_height:footer_y, 0:sidebar_width, :] = sidebar_color

        # Content region (central, x: 100-400)
        content_color = bg_color + self.rng.uniform(-20, 20, 3)
        content_color = np.clip(content_color, 0, 255)
        image[header_height:footer_y, sidebar_width:, :] = content_color

        # Add subtle noise (±5%) to each region
        noise = self.rng.uniform(-0.05 * 255, 0.05 * 255, image.shape)
        image = image + noise

        # Clip and convert to uint8
        image = np.clip(image, 0, 255).astype(np.uint8)

        return image

    def generate_dataset(self, count: int, output_dir: str) -> LabelsDict:
        """Generate count synthetic designs and save to disk.

        Args:
            count: Number of images to generate (e.g., 50)
            output_dir: Directory to save dataset/ (e.g., "./dataset")

        Returns:
            LabelsDict: {filename → ComponentDict} for all generated images

        Raises:
            OSError: If directory creation fails
            Exception: If image saving fails

        Side Effects:
            Creates output_dir/images/ directory
            Saves count PNG files: img_001.png, img_002.png, ..., img_050.png
            Saves output_dir/labels.json with ground-truth annotations

        Algorithm:
            1. Create output_dir/images/ directory
            2. For i in 0..count-1:
                a. Generate synthetic image via generate_synthetic_image(i)
                b. Save as PNG: output_dir/images/img_{i:03d}.png
                c. Call detect_layout_regions() to get ground-truth labels
                d. Store labels in dict
            3. Write labels.json: {filename → layout_dict}
            4. Return labels dict
        """
        output_path = Path(output_dir)
        images_dir = output_path / "images"
        images_dir.mkdir(parents=True, exist_ok=True)

        labels = {}

        for i in range(count):
            # Generate synthetic image
            image = self.generate_synthetic_image(i)

            # Save as PNG
            filename = f"img_{i+1:03d}.png"
            image_path = images_dir / filename
            Image.fromarray(image).save(image_path)

            # Get ground-truth labels via brightness detection
            region_dict = detect_layout_regions(image)
            labels[filename] = region_dict

        # Write labels.json
        labels_path = output_path / "labels.json"
        with open(labels_path, "w") as f:
            json.dump(labels, f, indent=2)

        return labels


def main():
    """CLI: python -m ane_design_model.dataset_generator --count 50 --output ./dataset"""
    parser = argparse.ArgumentParser(
        description="Generate synthetic design mockups with ground-truth labels"
    )
    parser.add_argument(
        "--count", type=int, default=50, help="Number of designs to generate"
    )
    parser.add_argument(
        "--output", type=str, default="./dataset", help="Output directory path"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )

    args = parser.parse_args()

    generator = DatasetGenerator(seed=args.seed)
    labels = generator.generate_dataset(args.count, args.output)

    print(f"Generated {args.count} designs in {args.output}/")
    print(f"Labels saved to {args.output}/labels.json")


if __name__ == "__main__":
    main()
