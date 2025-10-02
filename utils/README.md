# HNN Utils

This directory contains utility modules used throughout the HNN codebase.

## Modules

### `bilinear.py`
Model architectures for HNN experiments.

**Classes:**
- `Config`: Configuration dataclass for model parameters
- `Linear`: Simple linear transformation model
- `Bilinear`: Hadamard (element-wise multiplication) model with structure `head(left(x) * right(x))`
- `MLP`: Multi-layer perceptron with hidden layer

All models follow the same interface and can be swapped via configuration files.

### `muon.py`
Muon optimizer implementation with Newton-Schulz orthogonalization.

**Key features:**
- Momentum-based optimization
- Default learning rate: 0.02
- Designed for training neural networks efficiently

### `data_utils.py`
Data processing and training utilities.

**Functions:**
- `calculate_fvu()`: Compute Fraction of Variance Unexplained
- `load_model_and_tokenizer()`: Load LLM and tokenizer
- `create_pile_dataloader()`: Create streaming dataloader for Pile dataset
- `setup_optimizer()`: Initialize Muon or AdamW optimizer
- `normalize_data()`: Z-score normalization
- `compute_metrics()`: Calculate MSE, variance explained, and FVU
- `print_gpu_memory()`: Display GPU memory usage
- `save_model_checkpoint()`: Save model with metadata
- `plot_training_metrics()`: Generate training plots

### `__init__.py`
Package initialization file that exports all commonly used classes and functions.

## Usage

Import from the utils package:

```python
from utils import (
    Linear, Bilinear, MLP, Config,
    load_model_and_tokenizer,
    create_pile_dataloader,
    calculate_fvu
)
```

All imports are centralized through `__init__.py` for convenience.
