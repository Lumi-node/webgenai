# WebGenAI API Reference

This document provides a reference for the modules within the `ane_design_model` package.

---

## `ane_design_model/__init__.py`

This module typically serves as the package entry point, often exposing core components for easy access.

**Key Exports:**
*   (No specific classes/functions listed, usually imports core components from other modules.)

**Example Usage:**
```python
# Assuming core components are re-exported here
from ane_design_model import ModelTrainer, InferenceEngine
```

---

## `ane_design_model/benchmark.py`

Contains utilities and functions for evaluating the performance of the design models.

**Key Functions:**

### `evaluate_model(model, dataset, metrics)`
*   **Signature:** `evaluate_model(model: Model, dataset: Dataset, metrics: list[str]) -> dict`
*   **Description:** Runs a comprehensive evaluation of a trained model against a given dataset using specified performance metrics.
*   **Example Usage:**
    ```python
    from .model import Model
    from .dataset_generator import generate_test_set
    
    trained_model = load_trained_model()
    test_data = generate_test_set(size=100)
    results = evaluate_model(trained_model, test_data, ['accuracy', 'f1_score'])
    print(results)
    ```

---

## `ane_design_model/dataset_generator.py`

Provides tools for creating synthetic or loading real-world datasets for training and testing.

**Key Functions:**

### `generate_synthetic_data(config: dict) -> Dataset`
*   **Signature:** `generate_synthetic_data(config: dict) -> Dataset`
*   **Description:** Creates a synthetic dataset based on a provided configuration dictionary (e.g., number of samples, complexity level).
*   **Example Usage:**
    ```python
    from .dataset_generator import generate_synthetic_data
    
    config = {"num_samples": 500, "complexity": "medium"}
    synthetic_ds = generate_synthetic_data(config)
    print(f"Generated dataset size: {len(synthetic_ds)}")
    ```

### `load_dataset(path: str) -> Dataset`
*   **Signature:** `load_dataset(path: str) -> Dataset`
*   **Description:** Loads a dataset from a specified file path (e.g., CSV, JSON).
*   **Example Usage:**
    ```python
    from .dataset_generator import load_dataset
    
    real_ds = load_dataset("./data/real_designs.csv")
    ```

---

## `ane_design_model/inference.py`

Handles the process of using a trained model to make predictions on new, unseen data.

**Key Class:**

### `InferenceEngine`
*   **Signature:** `InferenceEngine(model: Model)`
*   **Description:** Manages the loading and execution of a trained model for real-time or batch inference.
*   **Methods:**
    *   `predict(input_data: np.ndarray) -> np.ndarray`: Generates predictions for the given input data.
    *   `predict_batch(input_data_list: list[np.ndarray]) -> list[np.ndarray]`: Processes a list of input data in a batch manner.
*   **Example Usage:**
    ```python
    from .model import Model
    from .inference import InferenceEngine
    
    trained_model = load_trained_model()
    engine = InferenceEngine(trained_model)
    
    new_input = np.random.rand(1, 10)
    predictions = engine.predict(new_input)
    print(f"Prediction result: {predictions}")
    ```

---

## `ane_design_model/ml_layout_detector.py`

Contains specialized algorithms for detecting and analyzing the layout structure within design inputs.

**Key Class:**

### `LayoutDetector`
*   **Signature:** `LayoutDetector(config: dict)`
*   **Description:** A utility class responsible for parsing raw design inputs (e.g., images or structured data) to identify key layout components.
*   **Methods:**
    *   `detect_components(design_input: Any) -> dict`: Returns a dictionary mapping detected components (e.g., 'component_A', 'boundary') to their bounding boxes or features.
    *   `normalize_layout(layout_data: dict) -> dict`: Standardizes the detected layout data into a format consumable by the main model.
*   **Example Usage:**
    ```python
    from .ml_layout_detector import LayoutDetector
    
    detector = LayoutDetector({"threshold": 0.7})
    raw_design = load_image("design_001.png")
    
    detected = detector.detect_components(raw_design)
    normalized = detector.normalize_layout(detected)
    ```

---

## `ane_design_model/model.py`

Defines the core structure and interface for the machine learning model itself.

**Key Class:**

### `Model`
*   **Signature:** `Model(architecture: str, hyperparameters: dict)`
*   **Description:** The abstract or concrete representation of the AI model. It holds the model weights and defines the forward pass logic.
*   **Methods:**
    *   `load_weights(path: str)`: Loads pre-trained weights from a file.
    *   `save_weights(path: str)`: Saves the current state of the model weights.
    *   `forward(input_tensor: torch.Tensor) -> torch.Tensor`: Executes the forward pass of the model.
*   **Example Usage:**
    ```python
    from .model import Model
    
    # Initialize a specific architecture (e.g., 'Transformer')
    design_model = Model(architecture="Transformer", hyperparameters={"dim": 256})
    design_model.load_weights("./weights/best_model.pth")
    
    # Assuming input_tensor is a PyTorch tensor
    output = design_model.forward(input_tensor)
    ```

---

## `ane_design_model/model_trainer.py`

Manages the entire training lifecycle, including optimization, loss calculation, and checkpointing.

**Key Class:**

### `ModelTrainer`
*   **Signature:** `ModelTrainer(model: Model, dataset: Dataset, optimizer_config: dict)`
*   **Description:** Orchestrates the training loop. It handles data loading, gradient computation, and weight updates.
*   **Methods:**
    *   `train(epochs: int, batch_size: int) -> TrainingHistory`: Executes the training process for a specified number of epochs. Returns a history object.
    *   `save_checkpoint(epoch: int, path: str)`: Saves the model state at a specific epoch.
*   **Example Usage:**
    ```python
    from .model import Model
    from .dataset_generator import load_dataset
    from .model_trainer import ModelTrainer
    
    model = Model(architecture="CNN", hyperparameters={})
    train_data = load_dataset("./data/training_set.csv")
    
    trainer = ModelTrainer(model, train_data, optimizer_config={"lr": 0.001})
    history = trainer.train(epochs=50, batch_size=32)
    
    trainer.save_checkpoint(50, "./checkpoints/final_run.pth")
    ```

---

## `tests/__init__.py`

(Testing module - No public API exposed for general use.)

---

## `tests/integration_test_model_trainer.py`

(Testing module - No public API exposed for general use.)

---

## `tests/unit_test_model_trainer.py`

(Testing module - No public API exposed for general use.)

---