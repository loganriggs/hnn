# HNN Experiments

This directory contains the main experimental scripts for HNN research.

## Scripts

### `transcoding.py`
Trains transcoder models to replicate MLP behavior within a single layer.

**Usage:**
```bash
python experiments/transcoding.py --config mlp_transcoding_debug.yaml
```

**What it does:**
- Extracts MLP input/output activations from a specified layer
- Trains a transcoder (Linear/Bilinear/MLP) to replicate the transformation
- Saves training metrics and model weights
- Generates training plots

### `layer_prediction.py`
Trains models to predict activations from one layer to another using token concatenation.

**Usage:**
```bash
python experiments/layer_prediction.py --config layer_pred_l3_l6_full.yaml
```

**What it does:**
- Extracts activations from input and target layers
- Concatenates multiple tokens to create richer input representations
- Trains a transcoder to predict target layer activations
- Saves training metrics and model weights
- Generates training plots

## Configuration

Both scripts use YAML configuration files from the `yaml_configs/` directory. See `yaml_configs/README.md` for details on creating and using configs.

## Output

- **Model weights**: Saved to `model_weights/`
- **Training plots**: Saved to `figures/`
- **Naming convention**: `{experiment_type}_{model_type}_{optimizer}_{n_batches}b.{ext}`

## Running Experiments

```bash
# Debug mode (20 batches)
python experiments/transcoding.py --config mlp_transcoding_debug.yaml

# Full training (1000 batches)
python experiments/transcoding.py --config default.yaml  # after setting debug: false

# Layer prediction
python experiments/layer_prediction.py --config layer_pred_l3_l6_full.yaml
```
