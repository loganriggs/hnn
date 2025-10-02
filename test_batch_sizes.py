# %%
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from model import Model, Config
from muon import Muon
import json

# %%
# Load Pythia-440m model and tokenizer
model_name = "EleutherAI/pythia-410m"
device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(model_name)
model.to(device)

layer_idx = 3

# %%
# Create dataloader for Pile dataset
dataset = load_dataset(
    "monology/pile-uncopyrighted",
    split="train",
    streaming=True
)

def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=128,
        padding="max_length",
        return_tensors="pt"
    )

tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text", "meta"])

# %%
# Test different batch sizes
results = []
batch_sizes = [64, 128, 256, 512, 1024, 2048, 4096]

print(f"Total GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
print(f"\nTesting batch sizes: {batch_sizes}\n")

for batch_size in batch_sizes:
    try:
        print(f"Testing batch_size={batch_size}...")

        # Clear cache
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        # Create dataloader
        pile_dataloader = DataLoader(tokenized_dataset, batch_size=batch_size)

        # Get activations
        activations = {}

        def get_mlp_input_hook(module, input, output):
            if 'mlp_inputs' not in activations:
                activations['mlp_inputs'] = []
            activations['mlp_inputs'].append(input[0].detach().cpu())

        def get_mlp_output_hook(module, input, output):
            if 'mlp_outputs' not in activations:
                activations['mlp_outputs'] = []
            activations['mlp_outputs'].append(output.detach().cpu())

        # Get first batch
        target_layer = model.gpt_neox.layers[layer_idx]
        hook_input = target_layer.mlp.register_forward_hook(get_mlp_input_hook)
        hook_output = target_layer.mlp.register_forward_hook(get_mlp_output_hook)

        batch = next(iter(pile_dataloader))
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)

        with torch.no_grad():
            model(input_ids=input_ids, attention_mask=attention_mask)

        hook_input.remove()
        hook_output.remove()

        # Stack activations
        mlp_inputs = torch.cat(activations['mlp_inputs'], dim=0)
        mlp_outputs = torch.cat(activations['mlp_outputs'], dim=0)
        d_model = mlp_inputs.shape[-1]

        # Initialize transcoder
        transcoder_cfg = Config(
            n_inputs=d_model,
            n_hidden=d_model,
            n_outputs=d_model,
            lr=0.001,
            device=device,
            bias=True
        )

        transcoder = Model(transcoder_cfg).to(device)
        muon_params = [p for p in transcoder.parameters() if p.ndim >= 2]
        adamw_params = [p for p in transcoder.parameters() if p.ndim < 2]
        optimizer = Muon(muon_params, lr=transcoder_cfg.lr, adamw_params=adamw_params)
        criterion = torch.nn.MSELoss()

        # Forward pass
        mlp_in_flat = mlp_inputs.reshape(-1, d_model).to(device)
        mlp_out_flat = mlp_outputs.reshape(-1, d_model).to(device)

        optimizer.zero_grad()
        transcoder_out = transcoder(mlp_in_flat)
        loss = criterion(transcoder_out, mlp_out_flat)
        loss.backward()
        optimizer.step()

        # Check memory
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        max_allocated = torch.cuda.max_memory_allocated() / 1024**3
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        free = total - allocated

        result = {
            'batch_size': batch_size,
            'success': True,
            'allocated_gb': round(allocated, 2),
            'reserved_gb': round(reserved, 2),
            'max_allocated_gb': round(max_allocated, 2),
            'free_gb': round(free, 2),
            'total_gb': round(total, 2),
            'loss': round(loss.item(), 6)
        }
        results.append(result)

        print(f"  ✓ Success: {allocated:.2f}GB allocated, {free:.2f}GB free, Loss: {loss.item():.6f}")

        # Clean up
        del transcoder, optimizer, mlp_in_flat, mlp_out_flat, mlp_inputs, mlp_outputs
        del input_ids, attention_mask, batch
        torch.cuda.empty_cache()

    except RuntimeError as e:
        if "out of memory" in str(e):
            print(f"  ✗ OOM Error at batch_size={batch_size}")
            result = {
                'batch_size': batch_size,
                'success': False,
                'error': 'OOM',
                'allocated_gb': round(torch.cuda.memory_allocated() / 1024**3, 2),
                'reserved_gb': round(torch.cuda.memory_reserved() / 1024**3, 2)
            }
            results.append(result)
            torch.cuda.empty_cache()
            break
        else:
            print(f"  ✗ Error: {e}")
            result = {
                'batch_size': batch_size,
                'success': False,
                'error': str(e)
            }
            results.append(result)
            break

# %%
# Save results
with open('batch_size_test_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print("\n" + "="*60)
print("SUMMARY")
print("="*60)
for r in results:
    if r['success']:
        print(f"batch_size={r['batch_size']:4d}: {r['allocated_gb']:.2f}GB allocated, {r['free_gb']:.2f}GB free")
    else:
        print(f"batch_size={r['batch_size']:4d}: FAILED - {r.get('error', 'Unknown')}")

# Find optimal batch size
successful = [r for r in results if r['success']]
if successful:
    optimal = max(successful, key=lambda x: x['batch_size'])
    print(f"\n✓ Optimal batch size: {optimal['batch_size']}")
    print(f"  Memory usage: {optimal['allocated_gb']:.2f}GB / {optimal['total_gb']:.2f}GB")
else:
    print("\n✗ No successful batch sizes")

print(f"\nResults saved to batch_size_test_results.json")
