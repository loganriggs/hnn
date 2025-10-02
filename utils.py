"""Utility functions for HNN transcoding experiments"""

import torch
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader


def calculate_fvu(x_orig, x_pred):
    """Calculate Fraction of Variance Unexplained"""
    mean = x_orig.mean(dim=0, keepdim=True)
    numerator = torch.mean(torch.sum((x_orig - x_pred)**2, dim=-1))
    denominator = torch.mean(torch.sum((x_orig - mean)**2, dim=-1))
    return numerator / (denominator + 1e-6)


def load_model_and_tokenizer(model_name="EleutherAI/pythia-410m", device="cuda"):
    """Load model and tokenizer"""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.to(device)
    model.eval()
    return model, tokenizer


def create_pile_dataloader(tokenizer, batch_size=512, max_length=128):
    """Create dataloader for Pile dataset with streaming"""
    dataset = load_dataset(
        "monology/pile-uncopyrighted",
        split="train",
        streaming=True
    )

    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_tensors="pt"
        )

    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["text", "meta"]
    )

    return DataLoader(tokenized_dataset, batch_size=batch_size)


def setup_optimizer(transcoder, optimizer_type="Muon", lr=0.001):
    """Setup optimizer (Muon or AdamW)"""
    if optimizer_type == "Muon":
        from muon import Muon
        all_params = list(transcoder.parameters())
        optimizer = Muon(all_params, lr=0.02, adamw_params=[])
    elif optimizer_type == "AdamW":
        optimizer = torch.optim.AdamW(transcoder.parameters(), lr=lr)
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")

    return optimizer


def normalize_data(data, mean=None, std=None):
    """Normalize data with z-score normalization"""
    if mean is None:
        mean = data.mean(0, keepdim=True)
    if std is None:
        std = data.std(0, keepdim=True) + 1e-6

    normalized = (data - mean) / std
    return normalized, mean, std


def compute_metrics(target, prediction):
    """Compute MSE, variance explained, and FVU"""
    # MSE
    mse = torch.mean((target - prediction)**2)

    # Variance explained
    residuals = target - prediction
    total_variance = torch.var(target)
    residual_variance = torch.var(residuals)
    var_explained = 1 - (residual_variance / total_variance)

    # FVU
    fvu = calculate_fvu(target, prediction)

    return {
        'mse': mse.item(),
        'variance_explained': var_explained.item(),
        'fvu': fvu.item()
    }


def plot_training_metrics(mse_losses, variance_explained, fvu_values,
                          title_suffix, save_path, use_log_scale=True):
    """Plot training metrics with optional log scale"""
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 4))

    # Plot MSE Loss
    ax1.plot(mse_losses)
    ax1.set_xlabel('Batch')
    ax1.set_ylabel('MSE Loss')
    ax1.set_title(f'MSE Loss ({title_suffix})')
    if use_log_scale:
        ax1.set_yscale('log')
        ax1.grid(True, which="both", ls="-", alpha=0.2)
    else:
        ax1.grid(True)

    # Plot Variance Explained
    ax2.plot(variance_explained)
    ax2.set_xlabel('Batch')
    ax2.set_ylabel('Variance Explained')
    ax2.set_title(f'Variance Explained ({title_suffix})')
    ax2.set_ylim([0, 1])
    ax2.grid(True)

    # Plot FVU
    ax3.plot(fvu_values)
    ax3.set_xlabel('Batch')
    ax3.set_ylabel('FVU (Fraction of Variance Unexplained)')
    ax3.set_title(f'FVU ({title_suffix})')
    if use_log_scale:
        ax3.set_yscale('log')
        ax3.grid(True, which="both", ls="-", alpha=0.2)
    else:
        ax3.grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Training plot saved to {save_path}")
    plt.close()


def print_gpu_memory(stage=""):
    """Print GPU memory usage"""
    if torch.cuda.is_available():
        print(f"\nGPU Memory {stage}:")
        print(f"  Allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        print(f"  Reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
        print(f"  Max allocated: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")


def save_model_checkpoint(transcoder, optimizer, save_path, **kwargs):
    """Save model checkpoint with metadata"""
    checkpoint = {
        'model_state_dict': transcoder.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        **kwargs
    }
    torch.save(checkpoint, save_path)
    print(f"Model saved to {save_path}")
