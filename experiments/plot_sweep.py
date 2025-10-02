#!/usr/bin/env python3
"""
Plot training metrics from a sweep of experiments.
Loads checkpoints and plots all curves on the same graph.
"""
import torch
import matplotlib.pyplot as plt
import argparse
import os
import sys
import glob

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

parser = argparse.ArgumentParser(description='Plot sweep results')
parser.add_argument('--pattern', type=str, required=True,
                    help='Glob pattern for checkpoint files (e.g., "transcoder_weights_bilinear_muon_20b_hidden_multiplier*.pt")')
parser.add_argument('--output', type=str, default=None,
                    help='Output filename (default: auto-generated from pattern)')
args = parser.parse_args()

# Find checkpoint files
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
weights_dir = os.path.join(project_root, 'model_weights')
pattern_path = os.path.join(weights_dir, args.pattern)

checkpoint_files = sorted(glob.glob(pattern_path))

if not checkpoint_files:
    print(f"No checkpoint files found matching: {pattern_path}")
    exit(1)

print(f"Found {len(checkpoint_files)} checkpoint files:")
for f in checkpoint_files:
    print(f"  - {os.path.basename(f)}")

# Load data from each checkpoint
results = []
for ckpt_path in checkpoint_files:
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)

    # Extract sweep parameter from filename
    filename = os.path.basename(ckpt_path)
    # Extract the sweep value (e.g., "hidden_multiplier0.25" -> "0.25x")
    if 'hidden_multiplier' in filename:
        param_str = filename.split('hidden_multiplier')[1].split('.pt')[0]
        label = f"{param_str}x"
    else:
        label = filename.split('_')[-1].replace('.pt', '')

    results.append({
        'label': label,
        'mse_losses': ckpt.get('mse_losses', []),
        'variance_explained': ckpt.get('variance_explained', []),
        'fvu_values': ckpt.get('fvu_values', []),
        'config': ckpt.get('config', {})
    })

# Sort by label (numerical sort for hidden_multiplier)
try:
    results.sort(key=lambda x: float(x['label'].replace('x', '')))
except:
    results.sort(key=lambda x: x['label'])

# Create plots
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Plot MSE Loss (log scale)
ax1 = axes[0]
for r in results:
    batch_numbers = list(range(len(r['mse_losses'])))
    ax1.plot(batch_numbers, r['mse_losses'], label=r['label'], linewidth=2)
ax1.set_xlabel('Batch', fontsize=12)
ax1.set_ylabel('MSE Loss', fontsize=12)
ax1.set_title('MSE Loss vs Batch', fontsize=14, fontweight='bold')
ax1.set_yscale('log')
ax1.grid(True, which="both", ls="-", alpha=0.2)
ax1.legend(title='Hidden Multiplier', fontsize=10)

# Plot Variance Explained
ax2 = axes[1]
for r in results:
    batch_numbers = list(range(len(r['variance_explained'])))
    ax2.plot(batch_numbers, r['variance_explained'], label=r['label'], linewidth=2)
ax2.set_xlabel('Batch', fontsize=12)
ax2.set_ylabel('Variance Explained', fontsize=12)
ax2.set_title('Variance Explained vs Batch', fontsize=14, fontweight='bold')
ax2.set_ylim([0, 1])
ax2.grid(True, alpha=0.3)
ax2.legend(title='Hidden Multiplier', fontsize=10)

# Plot FVU (log scale)
ax3 = axes[2]
for r in results:
    batch_numbers = list(range(len(r['fvu_values'])))
    ax3.plot(batch_numbers, r['fvu_values'], label=r['label'], linewidth=2)
ax3.set_xlabel('Batch', fontsize=12)
ax3.set_ylabel('FVU (Fraction of Variance Unexplained)', fontsize=12)
ax3.set_title('FVU vs Batch', fontsize=14, fontweight='bold')
ax3.set_yscale('log')
ax3.grid(True, which="both", ls="-", alpha=0.2)
ax3.legend(title='Hidden Multiplier', fontsize=10)

plt.tight_layout()

# Save figure
if args.output:
    output_path = args.output
else:
    # Auto-generate output filename
    output_name = args.pattern.replace('*.pt', '_sweep.png').replace('transcoder_weights_', 'sweep_')
    figures_dir = os.path.join(project_root, 'figures')
    os.makedirs(figures_dir, exist_ok=True)
    output_path = os.path.join(figures_dir, output_name)

plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"\nPlot saved to: {output_path}")

# Print final metrics summary
print("\nFinal Metrics Summary:")
print("=" * 70)
print(f"{'Multiplier':<15} {'MSE Loss':<15} {'Var Explained':<15} {'FVU':<15}")
print("=" * 70)
for r in results:
    mse_final = r['mse_losses'][-1] if r['mse_losses'] else 0
    ve_final = r['variance_explained'][-1] if r['variance_explained'] else 0
    fvu_final = r['fvu_values'][-1] if r['fvu_values'] else 0
    print(f"{r['label']:<15} {mse_final:<15.6f} {ve_final:<15.4f} {fvu_final:<15.6f}")
print("=" * 70)
