import torch
from torch import nn
from tqdm import tqdm
from dataclasses import dataclass, field
from typing import Optional
import itertools


@dataclass
class Config:
    """A configuration class for toy models."""
    n_inputs: int = 4
    n_hidden: int = 4
    n_outputs: int = 6

    lr: float = 0.01
    device: str = "cpu"
    bias: bool = False


class Bilinear(nn.Module):
    """A Hadamard neural network (bilinear) for transcoding representations."""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        # The bilinear (left/right) and head can be seen as an MLP.
        self.left = nn.Linear(self.cfg.n_inputs, self.cfg.n_hidden, bias=self.cfg.bias)
        self.right = nn.Linear(self.cfg.n_inputs, self.cfg.n_hidden, bias=self.cfg.bias)
        self.head = nn.Linear(self.cfg.n_hidden, self.cfg.n_outputs, bias=self.cfg.bias)

    @classmethod
    def from_config(cls, **kwargs):
        return cls(Config(**kwargs))

    @classmethod
    def from_pretrained(cls, path, device='cpu'):
        """Load a pretrained model from a checkpoint file."""
        torch.serialization.add_safe_globals([Config])
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        cfg = checkpoint['config']
        model = cls(cfg)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        return model, checkpoint

    @property
    def w_l(self):
        return self.left.weight

    @property
    def w_r(self):
        return self.right.weight

    @property
    def w_p(self):
        return self.head.weight

    def forward(self, x):
        return self.head(self.left(x) * self.right(x))


class Linear(nn.Module):
    """A simple linear model for transcoding representations."""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.linear = nn.Linear(self.cfg.n_inputs, self.cfg.n_outputs, bias=self.cfg.bias)

    @classmethod
    def from_config(cls, **kwargs):
        return cls(Config(**kwargs))

    @classmethod
    def from_pretrained(cls, path, device='cpu'):
        """Load a pretrained model from a checkpoint file."""
        torch.serialization.add_safe_globals([Config])
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        cfg = checkpoint['config']
        model = cls(cfg)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        return model, checkpoint

    def forward(self, x):
        return self.linear(x)


class MLP(nn.Module):
    """A standard MLP for transcoding representations."""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.fc1 = nn.Linear(self.cfg.n_inputs, self.cfg.n_hidden, bias=self.cfg.bias)
        self.fc2 = nn.Linear(self.cfg.n_hidden, self.cfg.n_outputs, bias=self.cfg.bias)

    @classmethod
    def from_config(cls, **kwargs):
        return cls(Config(**kwargs))

    @classmethod
    def from_pretrained(cls, path, device='cpu'):
        """Load a pretrained model from a checkpoint file."""
        torch.serialization.add_safe_globals([Config])
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        cfg = checkpoint['config']
        model = cls(cfg)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        return model, checkpoint

    def forward(self, x):
        return self.fc2(torch.nn.functional.gelu(self.fc1(x)))