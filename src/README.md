# Source Code Structure

This directory contains the main source code for the open-uncertainty project.

## Modules

### Training Scripts

- **head_temp_training.py**: Training script for temperature head calibration using transformer-based or Platt scaling approaches
- **scalar_temp_training.py**: Training script for scalar temperature optimization

### Utilities (utils/)

- **calibration_evaluation.py**: Functions for evaluating calibration metrics
- **calibration_losses.py**: Custom loss functions for calibration training
- **clustering.py**: Clustering utilities for semantic uncertainty
- **dsets.py**: Dataset loading and preprocessing
- **evaluation_metrics.py**: Metrics computation (ECE, AUROC, etc.)
- **text_utils.py**: Text processing utilities
- **utils.py**: General utility functions
- **process_best_results.py**: Results processing utilities

## Usage

### As a Python Module

After installation, you can import the modules:

```python
from src.utils import dsets
from src.utils.calibration_evaluation import evaluate_model_and_save_results
from src.utils.utils import set_seed
```

### Running Training Scripts

You can run the training scripts as Python modules:

```bash
# Run head temperature training
python -m src.head_temp_training --dset trivia_qa --model_name meta-llama/Llama-3.1-8B-Instruct

# Run scalar temperature training
python -m src.scalar_temp_training --dset trivia_qa --model_name meta-llama/Llama-3.1-8B-Instruct
```

Or after installation, use the console entry points:

```bash
# Run head temperature training
head-temp-training --dset trivia_qa --model_name meta-llama/Llama-3.1-8B-Instruct

# Run scalar temperature training
scalar-temp-training --dset trivia_qa --model_name meta-llama/Llama-3.1-8B-Instruct
```

## Installation

To install the package in development mode:

```bash
pip install -e .
```

This will make the `src` module available throughout your Python environment.
