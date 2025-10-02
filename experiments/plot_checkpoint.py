#!/usr/bin/env python3
"""
Plot training metrics from a single checkpoint file.
"""
import torch
import matplotlib.pyplot as plt
import argparse
import os
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

parser = argparse.ArgumentParser(description='Plot single checkpoint results')
parser.add_argument('--checkpoint', type=str, required=True,
                    help='Path to checkpoint file (relative to model_weights/)')
parser.add_argument('--output', type=str, default=None,
                    help='Output filename (default: auto-generated)')
args = parser.parse_args()

# Find checkpoint file
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
weights_dir = os.path.join(project_root, 'model_weights')

# Handle both full path and relative path
if os.path.isabs(args.checkpoint):
    checkpoint_path = args.checkpoint
elif os.path.exists(args.checkpoint):
    checkpoint_path = args.checkpoint
else:
    checkpoint_path = os.path.join(weights_dir, args.checkpoint)

if not os.path.exists(checkpoint_path):
    print(f"Checkpoint file not found: {checkpoint_path}")
    exit(1)

print(f"Loading checkpoint: {checkpoint_path}")

# Load checkpoint
ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

# Extract data
mse_losses = ckpt.get('mse_losses', [])
variance_explained = ckpt.get('variance_explained', [])
fvu_values = ckpt.get('fvu_values', [])

# Get metadata
config = ckpt.get('config', {})
filename = os.path.basename(checkpoint_path)

print(f"\nCheckpoint info:")
print(f"  MSE losses: {len(mse_losses)} batches")
print(f"  Final MSE: {mse_losses[-1]:.6f}" if mse_losses else "  No MSE data")
print(f"  Final Variance Explained: {variance_explained[-1]:.4f}" if variance_explained else "  No VE data")
print(f"  Final FVU: {fvu_values[-1]:.6f}" if fvu_values else "  No FVU data")

# Create plots
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Create x-axis (batch numbers from 0 to N-1)
batch_numbers = list(range(len(mse_losses)))

# Plot MSE Loss (log scale)
ax1 = axes[0]
ax1.plot(batch_numbers, mse_losses, linewidth=2, color='#1f77b4')
ax1.set_xlabel('Batch', fontsize=12)
ax1.set_ylabel('MSE Loss', fontsize=12)
ax1.set_title('MSE Loss vs Batch', fontsize=14, fontweight='bold')
ax1.set_yscale('log')
ax1.grid(True, which="both", ls="-", alpha=0.2)

# Plot Variance Explained
ax2 = axes[1]
ax2.plot(batch_numbers, variance_explained, linewidth=2, color='#ff7f0e')
ax2.set_xlabel('Batch', fontsize=12)
ax2.set_ylabel('Variance Explained', fontsize=12)
ax2.set_title('Variance Explained vs Batch', fontsize=14, fontweight='bold')
ax2.set_ylim([0, 1])
ax2.grid(True, alpha=0.3)

# Plot FVU (log scale)
ax3 = axes[2]
ax3.plot(batch_numbers, fvu_values, linewidth=2, color='#2ca02c')
ax3.set_xlabel('Batch', fontsize=12)
ax3.set_ylabel('FVU (Fraction of Variance Unexplained)', fontsize=12)
ax3.set_title('FVU vs Batch', fontsize=14, fontweight='bold')
ax3.set_yscale('log')
ax3.grid(True, which="both", ls="-", alpha=0.2)

plt.tight_layout()

# Save figure
if args.output:
    output_path = args.output
else:
    # Auto-generate output filename
    output_name = filename.replace('.pt', '.png')
    figures_dir = os.path.join(project_root, 'figures')
    os.makedirs(figures_dir, exist_ok=True)
    output_path = os.path.join(figures_dir, output_name)

plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"\nPlot saved to: {output_path}")
