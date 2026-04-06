# 🚀 WebGenAI Quick Start Guide

Welcome to WebGenAI! This guide will get you up and running quickly with the core functionalities of the `webgenai` package.

WebGenAI provides tools for generating, training, and evaluating AI models designed for web design tasks.

## Prerequisites

Ensure you have the package installed:

```bash
pip install webgenai
```

## Core Modules Overview

The package is structured around several key modules:

*   `ane_design_model/model.py`: Defines the core AI model architecture.
*   `ane_design_model/model_trainer.py`: Handles the training pipeline for the model.
*   `ane_design_model/dataset_generator.py`: Tools for creating synthetic or loading datasets.
*   `ane_design_model/inference.py`: Used to run predictions on the trained model.
*   `ane_design_model/ml_layout_detector.py`: Specific module for detecting elements within web layouts.

---

## 💡 Usage Examples

Here are a few practical examples demonstrating how to use the key components of WebGenAI.

### Example 1: Generating a Synthetic Dataset

Before training, you often need data. Use `dataset_generator.py` to create a sample dataset.

```python
from webgenai.ane_design_model.dataset_generator import DatasetGenerator

# Initialize the generator
generator = DatasetGenerator()

# Generate a small batch of synthetic web design data
print("--- Generating Synthetic Dataset ---")
synthetic_data = generator.generate_data(num_samples=10)

print(f"Successfully generated {len(synthetic_data)} samples.")
# In a real scenario, you would save this data to disk or a database
# print(synthetic_data[:2]) 
```

### Example 2: Training the Model

Once you have data, you can train the model using `model_trainer.py`.

*(Note: This example assumes you have a prepared dataset object or path.)*

```python
from webgenai.ane_design_model.model_trainer import ModelTrainer
# Assume 'training_data' is loaded from Example 1 or a file
# For this example, we'll use a placeholder.
training_data = "path/to/your/web_design_data.csv" 

# Initialize the trainer
trainer = ModelTrainer()

print("\n--- Starting Model Training ---")
# Train the model. This function handles loading the model, fitting, and saving.
trained_model_path = trainer.train(
    data_path=training_data, 
    epochs=10, 
    learning_rate=0.001
)

print(f"Training complete. Model saved to: {trained_model_path}")
```

### Example 3: Performing Inference and Layout Detection

After training, you use `inference.py` to get predictions. We can combine this with `ml_layout_detector.py` to see how the model interprets a new design input.

```python
from webgenai.ane_design_model.inference import ModelInferenceEngine
from webgenai.ane_design_model.ml_layout_detector import LayoutDetector

# 1. Load the trained model
MODEL_FILE = "path/to/your/trained_model.pth" # Use the path from Example 2
inference_engine = ModelInferenceEngine(model_path=MODEL_FILE)

# 2. Prepare input (e.g., an image or HTML snippet)
input_design_image = "path/to/new_web_design.png"

# 3. Run inference to get a high-level prediction
prediction = inference_engine.predict(input_design_image)
print("\n--- Model Prediction Result ---")
print(f"Overall design score: {prediction.get('score')}")

# 4. Use the Layout Detector to find specific elements
detector = LayoutDetector()
detected_elements = detector.detect_elements(input_design_image)

print("\n--- Detected Layout Elements ---")
for element in detected_elements:
    print(f"  - Type: {element['type']}, Bounding Box: {element['bbox']}")
```