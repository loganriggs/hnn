# HNN Evaluations

This directory contains evaluation and testing scripts for HNN models.

## Scripts

### `eval_check.py`
Comprehensive evaluation script for trained transcoder models.

**Usage:**
```bash
python evaluations/eval_check.py
```

**What it does:**
1. Loads the LLM (Pythia-410m) and Pile dataset
2. Prints sample datapoints with token IDs and decoded text
3. Computes baseline cross-entropy loss on the original model
4. Loads a trained transcoder checkpoint
5. Evaluates reconstruction quality (FVU, MSE, variance explained)
6. For MLP transcoders: computes CE loss with reconstructed activations injected
7. Saves all results to `eval_results/eval_results_{timestamp}.txt`

**Configuration:**
Edit the constants at the top of the script:
```python
MODEL_NAME = "EleutherAI/pythia-410m"
CHECKPOINT_PATH = "model_weights/transcoder_weights_mlp_muon_20b.pt"
N_EVAL_BATCHES = 10
BATCH_SIZE = 64
```

**Output:**
- Text file with comprehensive evaluation results
- Includes sample datapoints, baseline metrics, reconstruction quality, and downstream CE loss impact
- Saved to `eval_results/` directory with timestamp

## Example Results

For MLP transcoding (20 batches):
- Baseline CE loss: 2.8446
- Variance Explained: 30.07%
- FVU: 0.6993
- CE loss with reconstruction: 3.0062 (5.68% increase)

## Supported Checkpoint Types

- **MLP Transcoding**: Full evaluation including CE loss impact
- **Layer Prediction**: FVU and reconstruction metrics only (CE loss injection not yet implemented)

## Future Evaluations

Additional evaluation scripts can be added here for:
- Ablation studies
- Interpretability analysis
- Downstream task performance
- Scaling experiments
