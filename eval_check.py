"""
Evaluation script for HNN transcoder models.

This script:
1. Loads the LLM and dataset
2. Prints sample datapoints with tokens
3. Computes baseline CE loss
4. Loads a transcoder checkpoint
5. Evaluates FVU on the target task
6. Computes CE loss with reconstructed activations injected
7. Saves results to a text file
"""

import torch
import torch.nn.functional as F
from utils import load_model_and_tokenizer, create_pile_dataloader, calculate_fvu, normalize_data
from bilinear import Linear, Bilinear, MLP, Config
import json
from datetime import datetime


# Configuration
MODEL_NAME = "EleutherAI/pythia-410m"
CHECKPOINT_PATH = "model_weights/transcoder_weights_mlp_muon_20b.pt"  # Change this to evaluate different models
# Example options:
# - "model_weights/transcoder_weights_mlp_muon_20b.pt"
# - "model_weights/layer_pred_l3_l6_linear_muon_20b.pt"
CHECKPOINT_TYPE = "transcoding"  # "transcoding" or "layer_prediction"
N_EVAL_BATCHES = 10  # Number of batches to evaluate on
BATCH_SIZE = 64
MAX_SEQ_LENGTH = 128
N_SAMPLE_POSITIONS = 6  # Number of sequence positions to print
N_SAMPLE_DATAPOINTS = 3  # Number of datapoints to print

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}\n")

# Load model and tokenizer
print("=" * 80)
print("LOADING MODEL AND DATA")
print("=" * 80)
model, tokenizer = load_model_and_tokenizer(MODEL_NAME, device)
pile_dataloader = create_pile_dataloader(tokenizer, batch_size=BATCH_SIZE, max_length=MAX_SEQ_LENGTH)

print(f"Model: {MODEL_NAME}")
print(f"Total layers: {len(model.gpt_neox.layers)}")
print(f"Hidden dimension: {model.config.hidden_size}")

# Get first batch for inspection
first_batch = next(iter(pile_dataloader))
input_ids = first_batch['input_ids'].to(device)
attention_mask = first_batch['attention_mask'].to(device)

print("\n" + "=" * 80)
print("SAMPLE DATAPOINTS")
print("=" * 80)

# Print sample datapoints
for i in range(min(N_SAMPLE_DATAPOINTS, input_ids.shape[0])):
    print(f"\nDatapoint {i}:")
    print(f"{'Position':<10} {'Token ID':<10} {'Token'}")
    print("-" * 50)
    for pos in range(min(N_SAMPLE_POSITIONS, input_ids.shape[1])):
        token_id = input_ids[i, pos].item()
        token = tokenizer.decode([token_id])
        print(f"{pos:<10} {token_id:<10} {repr(token)}")

# Compute baseline CE loss
print("\n" + "=" * 80)
print("BASELINE CROSS-ENTROPY LOSS")
print("=" * 80)

model.eval()
total_ce_loss = 0.0
n_tokens = 0

with torch.no_grad():
    # First batch CE loss (detailed)
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits[:, :-1, :]  # Exclude last position
    targets = input_ids[:, 1:]  # Shift targets

    # Per-example CE loss for first few datapoints
    print(f"\nCE Loss for first {N_SAMPLE_DATAPOINTS} datapoints:")
    for i in range(min(N_SAMPLE_DATAPOINTS, logits.shape[0])):
        example_logits = logits[i].reshape(-1, logits.shape[-1])
        example_targets = targets[i].reshape(-1)
        example_loss = F.cross_entropy(example_logits, example_targets, reduction='mean')
        print(f"  Datapoint {i}: {example_loss.item():.4f}")

    # Overall first batch CE loss
    first_batch_loss = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        targets.reshape(-1),
        reduction='mean'
    )
    print(f"\nFirst batch average CE loss: {first_batch_loss.item():.4f}")

    total_ce_loss += first_batch_loss.item() * logits.shape[0] * logits.shape[1]
    n_tokens += logits.shape[0] * logits.shape[1]

# Compute CE loss over multiple batches
print(f"\nComputing CE loss over {N_EVAL_BATCHES} batches...")
batch_count = 1
for batch in pile_dataloader:
    if batch_count >= N_EVAL_BATCHES:
        break

    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits[:, :-1, :]
        targets = input_ids[:, 1:]

        batch_loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            targets.reshape(-1),
            reduction='mean'
        )

        total_ce_loss += batch_loss.item() * logits.shape[0] * logits.shape[1]
        n_tokens += logits.shape[0] * logits.shape[1]

    batch_count += 1

baseline_ce_loss = total_ce_loss / n_tokens
print(f"\nBaseline CE loss (averaged over {N_EVAL_BATCHES} batches): {baseline_ce_loss:.4f}")

# Load checkpoint
print("\n" + "=" * 80)
print("LOADING TRANSCODER CHECKPOINT")
print("=" * 80)

checkpoint = torch.load(CHECKPOINT_PATH, weights_only=False)
config = checkpoint['config']

print(f"Checkpoint: {CHECKPOINT_PATH}")
print(f"Config: {config}")

# Determine model type from config or filename
if "mlp" in CHECKPOINT_PATH.lower():
    model_type = "MLP"
elif "bilinear" in CHECKPOINT_PATH.lower():
    model_type = "Bilinear"
elif "linear" in CHECKPOINT_PATH.lower():
    model_type = "Linear"
else:
    # Infer from hidden size
    if config.n_hidden == config.n_inputs * 4:
        model_type = "MLP"
    else:
        model_type = "Linear"

print(f"Detected model type: {model_type}")

# Initialize transcoder
if model_type == "Linear":
    transcoder = Linear(config).to(device)
elif model_type == "Bilinear":
    transcoder = Bilinear(config).to(device)
elif model_type == "MLP":
    transcoder = MLP(config).to(device)

transcoder.load_state_dict(checkpoint['model_state_dict'])
transcoder.eval()

# Determine checkpoint type
if 'layer_idx' in checkpoint:
    CHECKPOINT_TYPE = "transcoding"
    layer_idx = checkpoint['layer_idx']
    print(f"Type: MLP Transcoding")
    print(f"Layer: {layer_idx}")
elif 'input_layer' in checkpoint:
    CHECKPOINT_TYPE = "layer_prediction"
    input_layer = checkpoint['input_layer']
    target_layer = checkpoint['target_layer']
    comb_seq_n = checkpoint['comb_seq_n']
    print(f"Type: Layer Prediction")
    print(f"Input layer: {input_layer}")
    print(f"Target layer: {target_layer}")
    print(f"Token concatenation: {comb_seq_n}")

# Evaluate FVU on target task
print("\n" + "=" * 80)
print("EVALUATING FVU ON TARGET TASK")
print("=" * 80)

total_fvu = 0.0
total_mse = 0.0
total_var_explained = 0.0
n_samples = 0

# Define hooks based on checkpoint type
activations = {}

if CHECKPOINT_TYPE == "transcoding":
    # MLP transcoding hooks
    def get_mlp_input_hook(module, input, output):
        if 'mlp_inputs' not in activations:
            activations['mlp_inputs'] = []
        activations['mlp_inputs'].append(input[0].detach().cpu())

    def get_mlp_output_hook(module, input, output):
        if 'mlp_outputs' not in activations:
            activations['mlp_outputs'] = []
        activations['mlp_outputs'].append(output.detach().cpu())

    pile_dataloader_eval = create_pile_dataloader(tokenizer, batch_size=BATCH_SIZE, max_length=MAX_SEQ_LENGTH)

    for batch_idx, batch in enumerate(pile_dataloader_eval):
        if batch_idx >= N_EVAL_BATCHES:
            break

        activations = {}
        target_layer_module = model.gpt_neox.layers[layer_idx]
        hook_input = target_layer_module.mlp.register_forward_hook(get_mlp_input_hook)
        hook_output = target_layer_module.mlp.register_forward_hook(get_mlp_output_hook)

        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)

        with torch.no_grad():
            model(input_ids=input_ids, attention_mask=attention_mask)

        hook_input.remove()
        hook_output.remove()

        mlp_inputs = torch.cat(activations['mlp_inputs'], dim=0)
        mlp_outputs = torch.cat(activations['mlp_outputs'], dim=0)

        # Flatten
        mlp_in_flat = mlp_inputs.reshape(-1, config.n_inputs).to(device)
        mlp_out_flat = mlp_outputs.reshape(-1, config.n_outputs).to(device)

        # Normalize
        in_norm, in_mean, in_std = normalize_data(mlp_in_flat)
        out_norm, out_mean, out_std = normalize_data(mlp_out_flat)

        # Predict
        with torch.no_grad():
            pred_norm = transcoder(in_norm)

        # Compute metrics
        mse = torch.mean((out_norm - pred_norm) ** 2)
        residuals = out_norm - pred_norm
        total_variance = torch.var(out_norm)
        residual_variance = torch.var(residuals)
        var_explained = 1 - (residual_variance / total_variance)
        fvu = calculate_fvu(out_norm, pred_norm)

        total_fvu += fvu.item() * mlp_in_flat.shape[0]
        total_mse += mse.item() * mlp_in_flat.shape[0]
        total_var_explained += var_explained.item() * mlp_in_flat.shape[0]
        n_samples += mlp_in_flat.shape[0]

elif CHECKPOINT_TYPE == "layer_prediction":
    # Layer prediction hooks
    def get_input_hook(module, input, output):
        if 'input_activations' not in activations:
            activations['input_activations'] = []
        activations['input_activations'].append(output[0].detach().cpu())

    def get_target_hook(module, input, output):
        if 'target_activations' not in activations:
            activations['target_activations'] = []
        activations['target_activations'].append(output[0].detach().cpu())

    pile_dataloader_eval = create_pile_dataloader(tokenizer, batch_size=BATCH_SIZE, max_length=MAX_SEQ_LENGTH)

    for batch_idx, batch in enumerate(pile_dataloader_eval):
        if batch_idx >= N_EVAL_BATCHES:
            break

        activations = {}
        input_layer_module = model.gpt_neox.layers[input_layer]
        target_layer_module = model.gpt_neox.layers[target_layer]
        hook_input = input_layer_module.register_forward_hook(get_input_hook)
        hook_target = target_layer_module.register_forward_hook(get_target_hook)

        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)

        with torch.no_grad():
            model(input_ids=input_ids, attention_mask=attention_mask)

        hook_input.remove()
        hook_target.remove()

        batch_input_acts = torch.cat(activations['input_activations'], dim=0)
        batch_target_acts = torch.cat(activations['target_activations'], dim=0)

        batch_size, seq_len, hidden = batch_input_acts.shape
        effective_seq = seq_len // comb_seq_n

        # Concatenate tokens
        input_concat = batch_input_acts[:, :effective_seq*comb_seq_n, :].reshape(
            batch_size, effective_seq, comb_seq_n * hidden
        ).to(device)

        target_acts = batch_target_acts[:, comb_seq_n-1::comb_seq_n, :][:, :effective_seq, :].to(device)

        # Flatten
        input_flat = input_concat.reshape(-1, config.n_inputs)
        target_flat = target_acts.reshape(-1, config.n_outputs)

        # Normalize
        input_norm, in_mean, in_std = normalize_data(input_flat)
        target_norm, out_mean, out_std = normalize_data(target_flat)

        # Predict
        with torch.no_grad():
            pred_norm = transcoder(input_norm)

        # Compute metrics
        mse = torch.mean((target_norm - pred_norm) ** 2)
        residuals = target_norm - pred_norm
        total_variance = torch.var(target_norm)
        residual_variance = torch.var(residuals)
        var_explained = 1 - (residual_variance / total_variance)
        fvu = calculate_fvu(target_norm, pred_norm)

        total_fvu += fvu.item() * input_flat.shape[0]
        total_mse += mse.item() * input_flat.shape[0]
        total_var_explained += var_explained.item() * input_flat.shape[0]
        n_samples += input_flat.shape[0]

avg_fvu = total_fvu / n_samples
avg_mse = total_mse / n_samples
avg_var_explained = total_var_explained / n_samples

print(f"\nResults over {N_EVAL_BATCHES} batches ({n_samples} samples):")
print(f"  Average FVU: {avg_fvu:.4f}")
print(f"  Average MSE: {avg_mse:.4f}")
print(f"  Average Variance Explained: {avg_var_explained:.4f}")

# Evaluate CE loss with reconstructed activations
print("\n" + "=" * 80)
print("EVALUATING CE LOSS WITH RECONSTRUCTED ACTIVATIONS")
print("=" * 80)

if CHECKPOINT_TYPE == "transcoding":
    print(f"\nInjecting reconstructed MLP outputs at layer {layer_idx}...")

    total_ce_loss_reconstructed = 0.0
    n_tokens_reconstructed = 0

    # Hook to replace MLP output with reconstruction
    def replace_mlp_output_hook(module, input, output):
        # Get MLP input and original output
        mlp_input = input[0]  # (batch, seq, hidden)
        mlp_output = output  # (batch, seq, hidden)
        original_shape = mlp_input.shape

        # Flatten
        mlp_in_flat = mlp_input.reshape(-1, config.n_inputs)
        mlp_out_flat = mlp_output.reshape(-1, config.n_outputs)

        # Normalize input and output (compute output stats for denormalization)
        in_norm, in_mean, in_std = normalize_data(mlp_in_flat)
        out_norm, out_mean, out_std = normalize_data(mlp_out_flat)

        # Predict normalized output
        with torch.no_grad():
            pred_norm = transcoder(in_norm)

        # Denormalize prediction
        pred_denorm = pred_norm * out_std + out_mean
        reconstructed = pred_denorm.reshape(original_shape)

        return reconstructed

    pile_dataloader_eval = create_pile_dataloader(tokenizer, batch_size=BATCH_SIZE, max_length=MAX_SEQ_LENGTH)

    for batch_idx, batch in enumerate(pile_dataloader_eval):
        if batch_idx >= N_EVAL_BATCHES:
            break

        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)

        # Register replacement hook
        target_layer_module = model.gpt_neox.layers[layer_idx]
        hook = target_layer_module.mlp.register_forward_hook(replace_mlp_output_hook)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits[:, :-1, :]
            targets = input_ids[:, 1:]

            batch_loss = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                targets.reshape(-1),
                reduction='mean'
            )

            total_ce_loss_reconstructed += batch_loss.item() * logits.shape[0] * logits.shape[1]
            n_tokens_reconstructed += logits.shape[0] * logits.shape[1]

        hook.remove()

    reconstructed_ce_loss = total_ce_loss_reconstructed / n_tokens_reconstructed
    ce_loss_increase = reconstructed_ce_loss - baseline_ce_loss

    print(f"\nCE loss with reconstructed activations: {reconstructed_ce_loss:.4f}")
    print(f"CE loss increase: {ce_loss_increase:.4f} ({ce_loss_increase/baseline_ce_loss*100:.2f}%)")

else:
    print("\nCE loss evaluation with reconstruction not implemented for layer prediction yet.")
    print("(This would require injecting layer activations, which is more complex)")
    reconstructed_ce_loss = None
    ce_loss_increase = None

# Save results
print("\n" + "=" * 80)
print("SAVING RESULTS")
print("=" * 80)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
results_file = f"eval_results/eval_results_{timestamp}.txt"

with open(results_file, 'w') as f:
    f.write("=" * 80 + "\n")
    f.write("HNN TRANSCODER EVALUATION RESULTS\n")
    f.write("=" * 80 + "\n\n")

    f.write(f"Timestamp: {timestamp}\n")
    f.write(f"Model: {MODEL_NAME}\n")
    f.write(f"Checkpoint: {CHECKPOINT_PATH}\n")
    f.write(f"Checkpoint type: {CHECKPOINT_TYPE}\n\n")

    if CHECKPOINT_TYPE == "transcoding":
        f.write(f"Layer: {layer_idx}\n")
    else:
        f.write(f"Input layer: {input_layer}\n")
        f.write(f"Target layer: {target_layer}\n")
        f.write(f"Token concatenation: {comb_seq_n}\n")

    f.write(f"\nModel type: {model_type}\n")
    f.write(f"Config: {config}\n\n")

    f.write("=" * 80 + "\n")
    f.write("SAMPLE DATAPOINTS\n")
    f.write("=" * 80 + "\n\n")

    first_batch = next(iter(create_pile_dataloader(tokenizer, batch_size=BATCH_SIZE, max_length=MAX_SEQ_LENGTH)))
    input_ids_sample = first_batch['input_ids']

    for i in range(min(N_SAMPLE_DATAPOINTS, input_ids_sample.shape[0])):
        f.write(f"Datapoint {i}:\n")
        f.write(f"{'Position':<10} {'Token ID':<10} {'Token'}\n")
        f.write("-" * 50 + "\n")
        for pos in range(min(N_SAMPLE_POSITIONS, input_ids_sample.shape[1])):
            token_id = input_ids_sample[i, pos].item()
            token = tokenizer.decode([token_id])
            f.write(f"{pos:<10} {token_id:<10} {repr(token)}\n")
        f.write("\n")

    f.write("=" * 80 + "\n")
    f.write("BASELINE METRICS\n")
    f.write("=" * 80 + "\n\n")

    f.write(f"Evaluation batches: {N_EVAL_BATCHES}\n")
    f.write(f"Baseline CE loss: {baseline_ce_loss:.4f}\n\n")

    f.write("=" * 80 + "\n")
    f.write("TRANSCODER RECONSTRUCTION METRICS\n")
    f.write("=" * 80 + "\n\n")

    f.write(f"Samples evaluated: {n_samples}\n")
    f.write(f"Average FVU: {avg_fvu:.4f}\n")
    f.write(f"Average MSE: {avg_mse:.4f}\n")
    f.write(f"Average Variance Explained: {avg_var_explained:.4f}\n\n")

    if reconstructed_ce_loss is not None:
        f.write("=" * 80 + "\n")
        f.write("CE LOSS WITH RECONSTRUCTED ACTIVATIONS\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"CE loss with reconstruction: {reconstructed_ce_loss:.4f}\n")
        f.write(f"CE loss increase: {ce_loss_increase:.4f} ({ce_loss_increase/baseline_ce_loss*100:.2f}%)\n")

print(f"\nResults saved to: {results_file}")
print("\n" + "=" * 80)
print("EVALUATION COMPLETE")
print("=" * 80)
