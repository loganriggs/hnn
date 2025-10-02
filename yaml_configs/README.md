# HNN Experiment Configurations

This folder contains YAML configuration files for different HNN experiments.

## Usage

Run experiments with specific configs using the `--config` flag:

```bash
# Transcoding experiments
python transcoding.py --config mlp_transcoding_debug.yaml
python transcoding.py --config default.yaml  # or omit --config for default

# Layer prediction experiments
python layer_prediction.py --config layer_pred_l3_l6_full.yaml
```

## Available Configs

- **default.yaml** - Default configuration (debug mode, 20 batches)
- **mlp_transcoding_debug.yaml** - MLP transcoding in debug mode
- **layer_pred_l3_l6_full.yaml** - Layer 3→6 prediction, full training (1000 batches)

## Creating New Configs

To create a new experiment configuration:

1. Copy an existing YAML file
2. Rename it descriptively (e.g., `bilinear_l5_l10_full.yaml`)
3. Modify the settings as needed
4. Run with `--config your_new_config.yaml`

## Config Structure

```yaml
model:
  name: "EleutherAI/pythia-410m"
  device: "cuda"

dataset:
  name: "monology/pile-uncopyrighted"
  split: "train"
  max_length: 128

transcoding:
  layer_idx: 3              # Which layer's MLP to transcode
  model_type: "MLP"         # "Linear", "Bilinear", or "MLP"
  optimizer_type: "Muon"    # "Muon" or "AdamW"
  batch_size: 512
  learning_rate: 0.001
  hidden_multiplier: 4      # Hidden = input_dim * multiplier
  bias: true
  debug: true               # true for quick tests, false for full runs
  n_batches: 20             # Number of batches in debug mode
  n_batches_full: 1000      # Number of batches in full mode

layer_prediction:
  input_layer: 3            # Source layer
  target_layer: 6           # Target layer
  comb_seq_n: 16            # Number of tokens to concatenate
  model_type: "Linear"
  optimizer_type: "Muon"
  batch_size: 64            # Smaller due to larger input dimension
  learning_rate: 0.001
  hidden_multiplier: 4
  bias: true
  debug: true
  n_batches: 20
  n_batches_full: 1000

evaluation:
  n_eval_batches: 10
  n_sample_positions: 6
  n_sample_datapoints: 3
```

## Tips

- Use descriptive filenames that indicate the experiment type and settings
- Keep `debug: true` for testing, set to `false` for production runs
- Muon optimizer will use lr=0.02 regardless of the `learning_rate` setting
- For layer prediction, smaller batch sizes are needed due to token concatenation
