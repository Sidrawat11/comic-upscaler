"""Shared model loading utility.

Loads RRDBNet weights from disk using position-based key mapping to handle
checkpoint naming mismatches. Called by both the profiler and the inference
engine so neither duplicates this logic.
"""

from pathlib import Path
from basicsr.archs.rrdbnet_arch import RRDBNet
import torch


def load_rrdbnet(model_path: Path) -> torch.nn.Module:
    """Builds RRDBNet, loads weights via position-based key mapping, and returns the model in eval mode on CUDA."""
    net = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
    loadnet = torch.load(model_path, map_location=torch.device('cpu'))

    # Unwrap common checkpoint wrappers
    if isinstance(loadnet, dict) and 'params_ema' in loadnet:
        weights = loadnet['params_ema']
    elif isinstance(loadnet, dict) and 'params' in loadnet:
        weights = loadnet['params']
    else:
        weights = loadnet

    model_state = net.state_dict()
    raw_keys = list(weights.keys())
    target_keys = list(model_state.keys())

    if len(raw_keys) != len(target_keys):
        print(f"Warning: Key count mismatch! File: {len(raw_keys)}, Model: {len(target_keys)}")

    new_state_dict = {}
    for i in range(min(len(raw_keys), len(target_keys))):
        new_state_dict[target_keys[i]] = weights[raw_keys[i]]

    net.load_state_dict(new_state_dict, strict=False)
    return net.eval().to('cuda').half()