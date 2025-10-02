#!/usr/bin/env python3
"""
Run a parameter sweep from a sweep configuration file.
"""
import yaml
import argparse
import subprocess
import sys
import os
import tempfile
import copy

parser = argparse.ArgumentParser(description='Run parameter sweep')
parser.add_argument('--config', type=str, required=True,
                    help='Sweep config file name in yaml_configs/ folder')
args = parser.parse_args()

# Load sweep configuration
config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'yaml_configs', args.config)
print(f"Loading sweep config from: {config_path}")
with open(config_path, 'r') as f:
    sweep_config = yaml.safe_load(f)

if 'sweep' not in sweep_config:
    print("ERROR: Config file does not contain a 'sweep' section")
    sys.exit(1)

sweep_param = sweep_config['sweep']['parameter']
sweep_values = sweep_config['sweep']['values']
base_config = sweep_config['config']

print(f"\nSweep parameter: {sweep_param}")
print(f"Sweep values: {sweep_values}")
print(f"\nRunning {len(sweep_values)} experiments\n")

# Run each sweep value
for idx, value in enumerate(sweep_values):
    print(f"{'='*70}")
    print(f"Experiment {idx + 1}/{len(sweep_values)}: {sweep_param} = {value}")
    print(f"{'='*70}\n")

    # Create a temporary config with this sweep value
    run_config = copy.deepcopy(base_config)

    # Set the sweep parameter (e.g., "transcoding.hidden_multiplier")
    param_parts = sweep_param.split('.')
    target = run_config
    for part in param_parts[:-1]:
        target = target[part]
    target[param_parts[-1]] = value

    # Add sweep metadata for filename generation
    run_config['_sweep_metadata'] = {
        'param': sweep_param,
        'value': value
    }

    # Write temporary config file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(run_config, f)
        temp_config_path = f.name

    try:
        # Run transcoding.py with this config
        result = subprocess.run(
            [sys.executable, 'transcoding.py', '--config-path', temp_config_path],
            cwd=os.path.dirname(__file__),
            check=True
        )

        print(f"\nCompleted {sweep_param} = {value}\n")

    except subprocess.CalledProcessError as e:
        print(f"ERROR: Experiment failed for {sweep_param} = {value}")
        print(f"Return code: {e.returncode}")
        sys.exit(1)
    finally:
        # Clean up temp file
        if os.path.exists(temp_config_path):
            os.unlink(temp_config_path)

print(f"\n{'='*70}")
print(f"Sweep complete! Ran {len(sweep_values)} experiments")
print(f"{'='*70}")
