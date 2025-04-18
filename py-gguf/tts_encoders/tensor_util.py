import torch.nn as nn
from torch.nn.utils.weight_norm import WeightNorm
from typing import Dict


def get_regularized_weight(modules: Dict[str, nn.Module], parameter_name: str) -> nn.Parameter:
    """
    Attempts to call torch the forward pre-hook in order to apply and assign weight normalization on
    a weight normalized nn.Module and return the normalized weight such that a GGUF compatible weight
    tensor can be determined on the fly.

    :param Dict[str, nn.Module] modules: a dictionary containing modules belonging to the current module context by name
    :param str parameter_name: the base parameter name from which the normalized weight derives.
    :return nn.Parameter: the computed normalized weight parameter.
    """
    assert "weight_g" in parameter_name or "weight_v" in parameter_name, f"Attempted to get the normalized weight for a non weight parameter, {parameter_name}."
    parent_module_name = ".".join(parameter_name.split(".")[:-1])
    if parent_module_name not in modules:
        raise KeyError(f"Failed to find module, {parent_module_name}, for parameter, {parameter_name}, in modules dictionary.")
    module = modules[parent_module_name]
    for hook in module._forward_pre_hooks.values():
        if isinstance(hook, WeightNorm):
            hook(module, None)
            break
    return module.weight
