#!/bin/bash
# Run bilinear expansion factor sweep

source hnn_venv/bin/activate

echo "===== Starting Bilinear Expansion Factor Sweep ====="
echo "Running 5 experiments with 500 batches each"
echo ""

for mult in 0.25 0.5 1 2 4; do
    echo "=================================================="
    echo "Running ${mult}x expansion..."
    echo "=================================================="
    python experiments/transcoding.py --config bilinear_sweep_${mult}x.yaml
    echo ""
    echo "Completed ${mult}x expansion"
    echo ""
done

echo "===== Sweep Complete ====="
