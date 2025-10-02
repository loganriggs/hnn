# %%
import torch
from torch.utils.data import DataLoader
from bilinear import Linear, Bilinear, MLP, Config
from utils import (
    calculate_fvu,
    load_model_and_tokenizer,
    create_pile_dataloader,
    setup_optimizer,
    normalize_data,
    compute_metrics,
    print_gpu_memory,
    save_model_checkpoint
)
# %%
# Load Pythia-410m model and tokenizer
model_name = "EleutherAI/pythia-410m"
device = "cuda" if torch.cuda.is_available() else "cpu"
model, tokenizer = load_model_and_tokenizer(model_name, device)

# Layer to extract activations from
layer_idx = 3

# %%
# Create dataloader for Pile dataset
pile_dataloader = create_pile_dataloader(tokenizer, batch_size=512, max_length=128)

# %%
# Define hooks for capturing activations
def get_mlp_input_hook(module, input, output):
    if 'mlp_inputs' not in activations:
        activations['mlp_inputs'] = []
    activations['mlp_inputs'].append(input[0].detach().cpu())

def get_mlp_output_hook(module, input, output):
    if 'mlp_outputs' not in activations:
        activations['mlp_outputs'] = []
    activations['mlp_outputs'].append(output.detach().cpu())

# Get first batch to determine d_model
activations = {}
target_layer = model.gpt_neox.layers[layer_idx]
hook_input = target_layer.mlp.register_forward_hook(get_mlp_input_hook)
hook_output = target_layer.mlp.register_forward_hook(get_mlp_output_hook)

first_batch = next(iter(pile_dataloader))
input_ids = first_batch['input_ids'].to(device)
attention_mask = first_batch['attention_mask'].to(device)

with torch.no_grad():
    model(input_ids=input_ids, attention_mask=attention_mask)

hook_input.remove()
hook_output.remove()

mlp_inputs = torch.cat(activations['mlp_inputs'], dim=0)
mlp_outputs = torch.cat(activations['mlp_outputs'], dim=0)
d_model = mlp_inputs.shape[-1]

print(f"MLP input shape: {mlp_inputs.shape}")
print(f"MLP output shape: {mlp_outputs.shape}")

# %%
# Initialize transcoder model to replicate MLP behavior
d_model = mlp_inputs.shape[-1]  # Get hidden dimension from MLP
transcoder_cfg = Config(
    n_inputs=d_model,
    n_hidden=d_model*4,
    n_outputs=d_model,
    lr=0.001,
    device=device,
    bias=True
)

# Choose model type: Linear, Bilinear, or MLP
MODEL_TYPE = "MLP"  # Change this to test different models
OPTIMIZER_TYPE = "Muon"  # "Muon" or "AdamW"
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
print(f"Using {OPTIMIZER_TYPE} optimizer")

# %%
# Training loop
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt

DEBUG = True  # Set to True for quick testing
n_batches = 20 if DEBUG else 1000  # Number of batches to train on

# Track metrics
mse_losses = []
variance_explained = []
fvu_values = []

transcoder.train()

# Check initial GPU memory
print_gpu_memory("before training")

pbar = tqdm(enumerate(pile_dataloader), total=n_batches, desc="Training")
for batch_idx, batch in pbar:
    if batch_idx >= n_batches:
        break

    # Clear activations for new batch
    activations = {}

    # Register hooks
    target_layer = model.gpt_neox.layers[layer_idx]
    hook_input = target_layer.mlp.register_forward_hook(get_mlp_input_hook)
    hook_output = target_layer.mlp.register_forward_hook(get_mlp_output_hook)

    # Get MLP activations
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)

    with torch.no_grad():
        model(input_ids=input_ids, attention_mask=attention_mask)

    # Remove hooks
    hook_input.remove()
    hook_output.remove()

    # Stack activations
    batch_mlp_inputs = torch.cat(activations['mlp_inputs'], dim=0)
    batch_mlp_outputs = torch.cat(activations['mlp_outputs'], dim=0)

    # Reshape: (batch, seq, hidden) -> (batch*seq, hidden)
    mlp_in_flat = batch_mlp_inputs.reshape(-1, d_model).to(device)
    mlp_out_flat = batch_mlp_outputs.reshape(-1, d_model).to(device)

    # Compute normalization stats for input and output separately
    in_mean = mlp_in_flat.mean(0, keepdim=True)
    in_std = mlp_in_flat.std(0, keepdim=True) + 1e-6
    out_mean = mlp_out_flat.mean(0, keepdim=True)
    out_std = mlp_out_flat.std(0, keepdim=True) + 1e-6

    # Normalize input and output
    mlp_in_norm = (mlp_in_flat - in_mean) / in_std
    mlp_out_norm = (mlp_out_flat - out_mean) / out_std

    # Forward pass through transcoder on normalized data
    optimizer.zero_grad()
    transcoder_out_norm = transcoder(mlp_in_norm)

    # Compute loss on normalized data
    loss = criterion(transcoder_out_norm, mlp_out_norm)

    # Compute variance explained on normalized data
    residuals = mlp_out_norm - transcoder_out_norm
    total_variance = torch.var(mlp_out_norm)
    residual_variance = torch.var(residuals)
    var_explained = 1 - (residual_variance / total_variance)

    # Compute FVU (Fraction of Variance Unexplained) on normalized data
    fvu = calculate_fvu(mlp_out_norm, transcoder_out_norm)

    # Store metrics
    mse_losses.append(loss.item())
    variance_explained.append(var_explained.item())
    fvu_values.append(fvu.item())

    # Backward pass
    loss.backward()
    optimizer.step()

    # Update progress bar with current metrics
    pbar.set_postfix({
        'MSE': f'{loss.item():.4f}',
        'VE': f'{var_explained.item():.3f}',
        'FVU': f'{fvu.item():.3f}'
    })

print("\nTraining complete!")

# Final GPU memory check
print_gpu_memory("after training")

# Plot metrics with log scale
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 4))

# Plot MSE Loss (log scale)
ax1.plot(mse_losses)
ax1.set_xlabel('Batch')
ax1.set_ylabel('MSE Loss')
ax1.set_title(f'MSE Loss ({MODEL_TYPE}, {OPTIMIZER_TYPE})')
ax1.set_yscale('log')
ax1.grid(True, which="both", ls="-", alpha=0.2)

# Plot Variance Explained
ax2.plot(variance_explained)
ax2.set_xlabel('Batch')
ax2.set_ylabel('Variance Explained')
ax2.set_title(f'Variance Explained ({MODEL_TYPE}, {OPTIMIZER_TYPE})')
ax2.set_ylim([0, 1])
ax2.grid(True)

# Plot FVU (log scale)
ax3.plot(fvu_values)
ax3.set_xlabel('Batch')
ax3.set_ylabel('FVU (Fraction of Variance Unexplained)')
ax3.set_title(f'FVU ({MODEL_TYPE}, {OPTIMIZER_TYPE})')
ax3.set_yscale('log')
ax3.grid(True, which="both", ls="-", alpha=0.2)

plt.tight_layout()
plot_path = f"figures/training_metrics_{MODEL_TYPE.lower()}_{OPTIMIZER_TYPE.lower()}_{n_batches}b.png"
plt.savefig(plot_path, dpi=150, bbox_inches='tight')
print(f"Training plot saved to {plot_path}")
plt.close()

# Save model weights
save_path = f"model_weights/transcoder_weights_{MODEL_TYPE.lower()}_{OPTIMIZER_TYPE.lower()}_{n_batches}b.pt"
save_model_checkpoint(
    transcoder,
    optimizer,
    save_path,
    config=transcoder_cfg,
    layer_idx=layer_idx,
    d_model=d_model,
    mse_losses=mse_losses,
    variance_explained=variance_explained,
    fvu_values=fvu_values
)

# %%
