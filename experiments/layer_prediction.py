# %%
import torch
import yaml
import argparse
import os
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt
from utils import (
    Linear, Bilinear, MLP, Config,
    calculate_fvu,
    load_model_and_tokenizer,
    create_pile_dataloader,
    setup_optimizer,
    normalize_data,
    compute_metrics,
    print_gpu_memory,
    save_model_checkpoint
)

# Parse command-line arguments
parser = argparse.ArgumentParser(description='HNN Layer Prediction Experiment')
parser.add_argument('--config', type=str, default='default.yaml',
                    help='Config file name in yaml_configs/ folder (default: default.yaml)')
args = parser.parse_args()

# Load configuration (go up one directory from experiments/)
config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'yaml_configs', args.config)
print(f"Loading config from: {config_path}")
with open(config_path, 'r') as f:
    config_dict = yaml.safe_load(f)


# %%
# Extract config values
INPUT_LAYER = config_dict['layer_prediction']['input_layer']
TARGET_LAYER = config_dict['layer_prediction']['target_layer']
COMB_SEQ_N = config_dict['layer_prediction']['comb_seq_n']
MAX_SEQ_LENGTH = config_dict['dataset']['max_length']
EFFECTIVE_SEQ_LENGTH = MAX_SEQ_LENGTH // COMB_SEQ_N
BATCH_SIZE = config_dict['layer_prediction']['batch_size']
MODEL_TYPE = config_dict['layer_prediction']['model_type']
OPTIMIZER_TYPE = config_dict['layer_prediction']['optimizer_type']
DEBUG = config_dict['layer_prediction']['debug']
n_batches = config_dict['layer_prediction']['n_batches'] if DEBUG else config_dict['layer_prediction']['n_batches_full']

print(f"\nConfiguration:")
print(f"  Input Layer: {INPUT_LAYER}")
print(f"  Target Layer: {TARGET_LAYER}")
print(f"  Token Concatenation: {COMB_SEQ_N} tokens")
print(f"  Effective Sequence Length: {EFFECTIVE_SEQ_LENGTH}")
print(f"  Batch Size: {BATCH_SIZE}")
print(f"  Model Type: {MODEL_TYPE}")
print(f"  Optimizer: {OPTIMIZER_TYPE}")
print(f"  Number of batches: {n_batches}")

# %%
# Load model and tokenizer from config
model_name = config_dict['model']['name']
device = config_dict['model']['device'] if torch.cuda.is_available() else "cpu"
model, tokenizer = load_model_and_tokenizer(model_name, device)

print(f"\nModel loaded: {model_name}")
print(f"Device: {device}")
print(f"Total layers: {len(model.gpt_neox.layers)}")

# %%
# Create dataloader for Pile dataset
pile_dataloader = create_pile_dataloader(tokenizer, batch_size=BATCH_SIZE, max_length=MAX_SEQ_LENGTH)

# %%
# Define hooks for capturing layer activations
activations = {}

def get_input_hook(module, input, output):
    if 'input_activations' not in activations:
        activations['input_activations'] = []
    # For GPTNeoX, we want the residual stream AFTER the layer
    # Output is the residual stream after this layer
    activations['input_activations'].append(output[0].detach().cpu())

def get_target_hook(module, input, output):
    if 'target_activations' not in activations:
        activations['target_activations'] = []
    activations['target_activations'].append(output[0].detach().cpu())

# Get first batch to determine dimensions
activations = {}
input_layer = model.gpt_neox.layers[INPUT_LAYER]
target_layer = model.gpt_neox.layers[TARGET_LAYER]
hook_input = input_layer.register_forward_hook(get_input_hook)
hook_target = target_layer.register_forward_hook(get_target_hook)

first_batch = next(iter(pile_dataloader))
input_ids = first_batch['input_ids'].to(device)
attention_mask = first_batch['attention_mask'].to(device)

with torch.no_grad():
    model(input_ids=input_ids, attention_mask=attention_mask)

hook_input.remove()
hook_target.remove()

input_acts = torch.cat(activations['input_activations'], dim=0)  # (batch, seq, hidden)
target_acts = torch.cat(activations['target_activations'], dim=0)

d_model = input_acts.shape[-1]
print(f"\nActivation shapes:")
print(f"  Input (layer {INPUT_LAYER}): {input_acts.shape}")
print(f"  Target (layer {TARGET_LAYER}): {target_acts.shape}")
print(f"  Hidden dimension: {d_model}")

# %%
# Initialize transcoder with concatenated input dimension
n_inputs = d_model * COMB_SEQ_N  # Concatenate COMB_SEQ_N tokens
n_outputs = d_model
hidden_multiplier = config_dict['layer_prediction']['hidden_multiplier']
learning_rate = config_dict['layer_prediction']['learning_rate']
bias = config_dict['layer_prediction']['bias']

transcoder_cfg = Config(
    n_inputs=n_inputs,
    n_hidden=n_inputs * hidden_multiplier,
    n_outputs=n_outputs,
    lr=learning_rate,
    device=device,
    bias=bias
)

if MODEL_TYPE == "Linear":
    transcoder = Linear(transcoder_cfg).to(device)
elif MODEL_TYPE == "Bilinear":
    transcoder = Bilinear(transcoder_cfg).to(device)
elif MODEL_TYPE == "MLP":
    transcoder = MLP(transcoder_cfg).to(device)
else:
    raise ValueError(f"Unknown model type: {MODEL_TYPE}")

# Select optimizer
optimizer = setup_optimizer(transcoder, optimizer_type=OPTIMIZER_TYPE, lr=transcoder_cfg.lr)

criterion = torch.nn.MSELoss()

print(f"\nTranscoder model initialized: {MODEL_TYPE}")
print(f"  Input dimension: {n_inputs} ({COMB_SEQ_N} tokens × {d_model})")
print(f"  Hidden dimension: {transcoder_cfg.n_hidden}")
print(f"  Output dimension: {n_outputs}")
print(f"Using {OPTIMIZER_TYPE} optimizer")

# %%
# Training loop
mse_losses = []
variance_explained = []
fvu_values = []

transcoder.train()

print_gpu_memory("before training")

pbar = tqdm(enumerate(pile_dataloader), total=n_batches, desc="Training")
for batch_idx, batch in pbar:
    if batch_idx >= n_batches:
        break

    # Clear activations
    activations = {}

    # Register hooks
    hook_input = input_layer.register_forward_hook(get_input_hook)
    hook_target = target_layer.register_forward_hook(get_target_hook)

    # Get activations
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)

    with torch.no_grad():
        model(input_ids=input_ids, attention_mask=attention_mask)

    hook_input.remove()
    hook_target.remove()

    # Stack activations
    batch_input_acts = torch.cat(activations['input_activations'], dim=0)  # (batch, seq, hidden)
    batch_target_acts = torch.cat(activations['target_activations'], dim=0)

    batch_size, seq_len, hidden = batch_input_acts.shape

    # Reshape for concatenation: group every COMB_SEQ_N tokens
    # (batch, seq, hidden) -> (batch, seq//COMB_SEQ_N, COMB_SEQ_N*hidden)
    effective_seq = seq_len // COMB_SEQ_N
    input_concat = batch_input_acts[:, :effective_seq*COMB_SEQ_N, :].reshape(
        batch_size, effective_seq, COMB_SEQ_N * hidden
    ).to(device)

    # Target: take every COMB_SEQ_N-th token (or mean, or last token of each group)
    # Using last token of each group for simplicity
    target_acts = batch_target_acts[:, COMB_SEQ_N-1::COMB_SEQ_N, :][:, :effective_seq, :].to(device)

    # Flatten: (batch, effective_seq, dim) -> (batch*effective_seq, dim)
    input_flat = input_concat.reshape(-1, n_inputs)
    target_flat = target_acts.reshape(-1, n_outputs)

    # Normalize
    in_mean = input_flat.mean(0, keepdim=True)
    in_std = input_flat.std(0, keepdim=True) + 1e-6
    out_mean = target_flat.mean(0, keepdim=True)
    out_std = target_flat.std(0, keepdim=True) + 1e-6

    input_norm = (input_flat - in_mean) / in_std
    target_norm = (target_flat - out_mean) / out_std

    # Forward pass
    optimizer.zero_grad()
    pred_norm = transcoder(input_norm)

    # Compute loss
    loss = criterion(pred_norm, target_norm)

    # Compute metrics
    residuals = target_norm - pred_norm
    total_variance = torch.var(target_norm)
    residual_variance = torch.var(residuals)
    var_explained = 1 - (residual_variance / total_variance)

    fvu = calculate_fvu(target_norm, pred_norm)

    # Store metrics
    mse_losses.append(loss.item())
    variance_explained.append(var_explained.item())
    fvu_values.append(fvu.item())

    # Backward pass
    loss.backward()
    optimizer.step()

    # Update progress bar
    pbar.set_postfix({
        'MSE': f'{loss.item():.4f}',
        'VE': f'{var_explained.item():.3f}',
        'FVU': f'{fvu.item():.3f}'
    })

print("\nTraining complete!")

print_gpu_memory("after training")

# %%
# Plot metrics with log scale
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 4))

# Plot MSE Loss (log scale)
ax1.plot(mse_losses)
ax1.set_xlabel('Batch')
ax1.set_ylabel('MSE Loss')
ax1.set_title(f'MSE Loss ({MODEL_TYPE}, {OPTIMIZER_TYPE}, L{INPUT_LAYER}→L{TARGET_LAYER})')
ax1.set_yscale('log')
ax1.grid(True, which="both", ls="-", alpha=0.2)

# Plot Variance Explained
ax2.plot(variance_explained)
ax2.set_xlabel('Batch')
ax2.set_ylabel('Variance Explained')
ax2.set_title(f'Variance Explained ({MODEL_TYPE}, {OPTIMIZER_TYPE}, L{INPUT_LAYER}→L{TARGET_LAYER})')
ax2.set_ylim([0, 1])
ax2.grid(True)

# Plot FVU (log scale)
ax3.plot(fvu_values)
ax3.set_xlabel('Batch')
ax3.set_ylabel('FVU')
ax3.set_title(f'FVU ({MODEL_TYPE}, {OPTIMIZER_TYPE}, L{INPUT_LAYER}→L{TARGET_LAYER})')
ax3.set_yscale('log')
ax3.grid(True, which="both", ls="-", alpha=0.2)

plt.tight_layout()
plot_path = f"figures/layer_pred_l{INPUT_LAYER}_l{TARGET_LAYER}_{MODEL_TYPE.lower()}_{OPTIMIZER_TYPE.lower()}_{n_batches}b.png"
plt.savefig(plot_path, dpi=150, bbox_inches='tight')
print(f"Training plot saved to {plot_path}")
plt.close()

# Save model weights
save_path = f"model_weights/layer_pred_l{INPUT_LAYER}_l{TARGET_LAYER}_{MODEL_TYPE.lower()}_{OPTIMIZER_TYPE.lower()}_{n_batches}b.pt"
save_model_checkpoint(
    transcoder,
    optimizer,
    save_path,
    config=transcoder_cfg,
    input_layer=INPUT_LAYER,
    target_layer=TARGET_LAYER,
    comb_seq_n=COMB_SEQ_N,
    d_model=d_model,
    mse_losses=mse_losses,
    variance_explained=variance_explained,
    fvu_values=fvu_values
)

# %%
