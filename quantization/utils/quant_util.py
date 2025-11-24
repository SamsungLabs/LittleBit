import importlib
import re
from collections import ChainMap
from functools import partial
from typing import Dict, List, Optional, Tuple

import torch.nn as nn

from quantization.modules import LittleBitLinear

__all__ = ['patch_inst']

# Global member of the file that contains the mapping of quantized modules
_DEFAULT_MAPPING = {
    nn.Linear: LittleBitLinear,
}


def _match_pattern(patterns: list, model: nn.Module, name: str, mod: nn.Module) -> bool:
    """
    Match a module against a list of patterns.

    Args:
        patterns (list): List of patterns to match against.
        model (nn.Module): Model to search for the module.
        name (str): Name of the module.
        mod (nn.Module): Module to match.

    Returns:
        bool: True if the module matches any of the patterns, False otherwise.
    """
    for _pattern in patterns:
        if type(_pattern) == str:
            _cond = partial(lambda _module_name, _prefix: is_module_include(_module_name, _prefix), _prefix=_pattern)
        elif type(_pattern) == type:
            _cond = partial(lambda _module_name, _cls_type: type(mod) == _cls_type, _cls_type=_pattern)
        elif type(_pattern) == re.Pattern:
            _cond = partial(lambda _module_name, _regex: bool(_regex.search(_module_name)), _regex=_pattern)
        else:
            raise ValueError(f'Invalid pattern: {_pattern}')
        if _cond(name):
            return True
    return False


def patch_inst(
    model: nn.Module,
    mapping: Optional[Dict[type, type]] = _DEFAULT_MAPPING,
    convert_kwargs: Optional[List[Tuple[list, dict]]] = None,
    exclude_names: Optional[List[str]] = None,
    device_map: dict = None,
):
    """
    Patch instances of a model with quantized modules.

    Args:
        model (nn.Module): Model to patch.
        mapping (Optional[Dict[type, type]]): Mapping of original module types to quantized module types.
        convert_kwargs (Optional[List[Tuple[list, dict]]]): List of tuples containing patterns and keyword
            arguments for converting modules to quantized modules.
        exclude_names (Optional[List[str]]): List of layer names that should not be converted.
        device_map (dict): A dictionary mapping device names to devices.
    """
    convert_kwargs = convert_kwargs or []
    exclude_names = exclude_names or []
    device_map = device_map or {}

    if device_map == "auto":
        from accelerate import infer_auto_device_map
        device_map = infer_auto_device_map(model)
    default_device = device_map.get("", "cpu")

    mapping_chained = ChainMap({}, mapping)

    # assume model is tree structure
    # HACK: depends on the implementation detail of pytorch that .named_modules() yields modules in preordering
    for name, mod in model.named_modules():
        if name in exclude_names:
            continue

        convert_kwargs_ = {}
        for pattern, d in convert_kwargs:
            if _match_pattern(pattern, model, name, mod):
                convert_kwargs_.update(d)

        if type(mod) in mapping_chained:
            mod.__class__ = mapping_chained[type(mod)]

        if hasattr(mod, '__quant_convert__'):
            if len(device_map) == 0:
                mod.__quant_convert__(**convert_kwargs_)
            else:
                mod.to(default_device)
                mod.__quant_convert__(**convert_kwargs_)
                mod.to("cpu") if default_device != "cpu" else None


def is_module_include(name, target):
    """
    Check if a module name includes a target substring.

    Args:
        name (str): Full name of the module.
        target (str): Target substring to check for inclusion.

    Returns:
        bool: True if the module name includes the target substring, False otherwise.
    """
    _module_names = name.split(".")
    while len(_module_names) > 0:
        if ".".join(_module_names).startswith(target):
            return True
        _module_names.pop(0)
    return False


def load_module_and_get_attr(package_path, module_name):
    """
    Load a module from a package and get a specific attribute.

    Args:
        package_path (str): The path to the package.
        module_name (str): The name of the module.
        attr_name (str): The name of the attribute.

    Returns:
        attr: The attribute from the module.
    """
    package = importlib.import_module(package_path)

    module = getattr(package, module_name)
    if module is None:
        raise ValueError(f"{module_name} not found in {package}.")

    return module


def get_quant_func_and_mod(quant_func_name, quant_mod_name):
    """
    Get a quant function and a quant module.

    Args:
        quant_func_name (str): The name of the quant function.
        quant_mod_name (str): The name of the quant module.

    Returns:
        quant_func: The quant function.
        quant_mod: The quant module.
    """
    for name in (quant_func_name, quant_mod_name):
        if not isinstance(name, str):
            raise ValueError("All names must be strings.")

    quant_func_package = "quantization.functions"
    quant_mod_package = "quantization.modules"

    quant_func = load_module_and_get_attr(quant_func_package, quant_func_name)
    quant_mod = load_module_and_get_attr(quant_mod_package, quant_mod_name)

    return quant_func, quant_mod
