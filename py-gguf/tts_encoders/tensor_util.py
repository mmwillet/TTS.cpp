import torch
import torch.nn as nn
from torch.nn.utils.parametrizations import _WeightNorm
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


def get_normalized_weight_from_parametrizations(modules: Dict[str, nn.Module], parameter_name: str) -> torch.Tensor:
    """
    Attempts to call the default parametrization forward pass for weight normalization such that the true weight
    can be determined via the stored parametrized variables.

    :param Dict[str, nn.Module] modules: a dictionary containing modules belonging to the current module context by name
    :param str parameter_name: the base parameter name from which the normalized weight is to be derived.
    :return torch.Tensor: the computed normalized weight tensor.
    """
    parent_module_name = parameter_name.split(".parametrizations")[0]
    if parent_module_name not in modules:
        raise KeyError(f"Failed to find module, {parent_module_name}, for parameter, {parameter_name}, in modules dictionary.")
    module = modules[parent_module_name]
    if "weight" not in module.parametrizations:
        raise KeyError(f"Failed to find parameterized weight on module, {parent_module_name}, for parameter, {parameter_name}.")
    assert isinstance(module.parametrizations["weight"][0], _WeightNorm)
    return torch._weight_norm(
        module.parametrizations["weight"].original1,
        module.parametrizations["weight"].original0,
        module.parametrizations["weight"][0].dim
    )
