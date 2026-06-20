"""Checkpoint helpers for F2R.

This module keeps checkpoint loading consistent across training and validation
scripts. It also strips common ``module.`` prefixes produced by
DistributedDataParallel checkpoints.
"""

from __future__ import annotations

import os
from collections import OrderedDict
from typing import Any, Mapping, Optional

import torch

PWCNET_URL = "http://content.sniklaus.com/github/pytorch-pwc/network-default.pytorch"


def torch_load(path: str, map_location: str | torch.device = "cpu") -> Any:
    """Load a PyTorch checkpoint while remaining compatible with older PyTorch.

    PyTorch 2.x supports ``weights_only=True``. Some older releases do not, so
    we fall back to the legacy call when necessary.
    """
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)


def get_state_dict(checkpoint: Any) -> Mapping[str, torch.Tensor]:
    """Return the state-dict from a raw state-dict or checkpoint dictionary."""
    if isinstance(checkpoint, Mapping):
        for key in ("state_dict", "model", "params", "net"):
            value = checkpoint.get(key)
            if isinstance(value, Mapping):
                return value
        return checkpoint
    raise TypeError(f"Unsupported checkpoint type: {type(checkpoint)!r}")


def clean_state_dict(state_dict: Mapping[str, torch.Tensor]) -> OrderedDict:
    """Remove common wrapper prefixes from checkpoint keys."""
    cleaned = OrderedDict()
    for key, value in state_dict.items():
        if key.startswith("module."):
            key = key[len("module."):]
        cleaned[key] = value
    return cleaned


def load_model_weights(model: torch.nn.Module, path: str, *, map_location="cpu", strict: bool = True):
    """Load weights into ``model`` and return the ``load_state_dict`` result."""
    checkpoint = torch_load(path, map_location=map_location)
    state_dict = clean_state_dict(get_state_dict(checkpoint))
    return model.load_state_dict(state_dict, strict=strict)


def load_pwcnet_weights(model: torch.nn.Module, path: Optional[str] = None, *, map_location="cpu"):
    """Load PWC-Net weights from a local file or the official public URL.

    The original PWC-Net checkpoint uses keys beginning with ``module`` while
    this repository's implementation expects ``net``. The conversion below
    follows the common PyTorch-PWC loading convention.
    """
    if path and os.path.isfile(path):
        checkpoint = torch_load(path, map_location=map_location)
        state_dict = get_state_dict(checkpoint)
    else:
        state_dict = torch.hub.load_state_dict_from_url(
            url=PWCNET_URL,
            file_name="pwc-default",
            map_location=map_location,
            progress=True,
        )

    converted = OrderedDict()
    for key, value in state_dict.items():
        if key.startswith("module"):
            key = key.replace("module", "net", 1)
        if key.startswith("module."):
            key = key[len("module."):]
        converted[key] = value
    return model.load_state_dict(converted)
