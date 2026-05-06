import importlib
import os
import gc
import json
import re
import gc
from collections import ChainMap, defaultdict
from functools import partial
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from safetensors import safe_open
from transformers import AutoConfig, AutoModelForCausalLM

from quantization.modules import LittleBitLinear

__all__ = ['patch_inst', 'load_quantized_model', 'apply_littlebit_patch', 'get_quant_func_and_mod']


def _match_pattern(patterns: list, model: nn.Module, name: str, mod: nn.Module) -> bool:
    """Match a module against a list of patterns (str, type, or regex)."""
    for _pattern in patterns:
        if isinstance(_pattern, str):
            if _pattern in name:
                return True
        elif isinstance(_pattern, type):
            if isinstance(mod, _pattern):
                return True
        elif isinstance(_pattern, re.Pattern):
            if _pattern.search(name):
                return True
    return False


def patch_inst(
    model: nn.Module,
    mapping: Optional[Dict[type, type]] = None,
    convert_kwargs: Optional[List[Tuple[list, dict]]] = None,
    exclude_names: Optional[List[str]] = None,
    device_map: dict = None,
):
    """
    Recursively patch instances of a model with quantized modules.
    """
    if mapping is None:
        from quantization.modules import LittleBitLinear
        mapping = {nn.Linear: LittleBitLinear}

    convert_kwargs = convert_kwargs or []
    exclude_names = exclude_names or []
    device_map = device_map or {}

    mapping_chained = ChainMap({}, mapping)
    default_device = device_map.get("", "cpu") if device_map else "cpu"

    for name, mod in model.named_modules():
        if any(ex in name for ex in exclude_names):
            continue

        current_kwargs = {}
        for pattern, d in convert_kwargs:
            if _match_pattern(pattern, model, name, mod):
                current_kwargs.update(d)

        if type(mod) in mapping_chained:
            mod.__class__ = mapping_chained[type(mod)]

            if hasattr(mod, '__quant_convert__'):
                original_device = next(mod.parameters(), torch.device("cpu")).device if list(
                    mod.parameters()) else default_device

                if device_map and original_device != default_device:
                    mod.to(default_device)

                mod.__quant_convert__(**current_kwargs)

                if device_map and original_device != default_device:
                    mod.to(original_device)


def load_module_and_get_attr(package_path, module_name):
    """
    Load a module from a package and get a specific attribute.
    """
    try:
        package = importlib.import_module(package_path)
        module = getattr(package, module_name)
    except (ImportError, AttributeError) as e:
        raise ValueError(f"Could not load {module_name} from {package_path}: {e}")
    return module


def get_quant_func_and_mod(quant_func_name, quant_mod_name):
    """
    Get a quant function and a quant module.
    """
    for name in (quant_func_name, quant_mod_name):
        if not isinstance(name, str):
            raise ValueError("All names must be strings.")

    quant_func_package = "quantization.functions"
    quant_mod_package = "quantization.modules"

    quant_func = load_module_and_get_attr(quant_func_package, quant_func_name)
    quant_mod = load_module_and_get_attr(quant_mod_package, quant_mod_name)

    return quant_func, quant_mod


def apply_littlebit_patch(model: nn.Module, args, do_train: bool = False):
    """
    A high-level wrapper to apply LittleBit quantization patch to any model.
    This unifies the logic used in main.py (training) and eval.py (inference).
    
    Args:
        model: The model to patch (e.g., from AutoModel.from_config).
        args: Namespace containing quantization arguments (quant_func, split_dim, etc.).
        do_train: Whether to initialize for training (SVD decomp) or inference (empty/loading).
    """
    quant_func_name = getattr(args, "quant_func", "STEBinary")
    quant_mod_name = getattr(args, "quant_mod", "LittleBitLinear")
    quant_func, quant_mod = get_quant_func_and_mod(quant_func_name, quant_mod_name)

    mapping = {nn.Linear: quant_mod}

    common_kwargs = {
        "do_train": do_train,
        "quant_func": quant_func,
        "residual": getattr(args, "residual", False),
        "split_dim": getattr(args, "split_dim", 1024),
        "eff_bit": getattr(args, "eff_bit", 1.0),
        "min_split_dim": getattr(args, "min_split_dim", 8),
        "use_itq": getattr(args, "use_itq", False),
        "itq_n_iter": getattr(args, "itq_n_iter", 50),
    }

    KV_PATTERN = [re.compile(r'\.k_proj$'), re.compile(r'\.v_proj$')]
    kv_kwargs = {
        "ratio_factor": getattr(args, "kv_factor", 1.0),
    }

    convert_kwargs = [
        ([nn.Linear], common_kwargs),
        (KV_PATTERN, kv_kwargs),
    ]

    config = getattr(model, "config", None)
    model_type = getattr(config, "model_type", "").lower() if config else ""
    model_id_str = getattr(args, "model_id", "").lower()

    if "phi" in model_type or "phi" in model_id_str:
        try:
            from transformers.models.phi3.modeling_phi3 import Phi3Attention
            from quantization.modules.attention import PhiQKVSplitAttention

            mapping.update({Phi3Attention: PhiQKVSplitAttention})
            convert_kwargs.append(([Phi3Attention], {'config': config}))
        except ImportError:
            pass

    patch_inst(
        model,
        mapping=mapping,
        convert_kwargs=convert_kwargs,
        exclude_names=["lm_head"],
    )

    return model


# ===== Section 3: Model Loading & State Dict Processing =====
def _load_and_process_state_dict(model_path: str, torch_dtype: torch.dtype):
    """
    Loads state_dict, handling safetensors (single or sharded) and unpacking binary weights.
    Returns (state_dict, was_packed_boolean).
    """
    state_dict = {}

    # 1. Load raw tensors
    index_path = os.path.join(model_path, "model.safetensors.index.json")
    single_path = os.path.join(model_path, "model.safetensors")

    if os.path.exists(index_path):
        with open(index_path, "r", encoding="utf-8") as f:
            index = json.load(f)
        # Load unique shard files
        shard_files = set(index["weight_map"].values())
        for shard_file in shard_files:
            with safe_open(os.path.join(model_path, shard_file), framework="pt", device="cpu") as f:
                for key in f.keys():
                    state_dict[key] = f.get_tensor(key)
    elif os.path.exists(single_path):
        with safe_open(single_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                state_dict[key] = f.get_tensor(key)
    else:
        # Fallback to pytorch bin if needed, or raise error
        bin_path = os.path.join(model_path, "pytorch_model.bin")
        if os.path.exists(bin_path):
            import inspect
            sig = inspect.signature(torch.load)
            kwargs = {"map_location": "cpu"}
            if "mmap" in sig.parameters: kwargs["mmap"] = True
            if "weights_only" in sig.parameters: kwargs["weights_only"] = True

            try:
                state_dict = torch.load(bin_path, **kwargs)
            except Exception:
                kwargs["weights_only"] = False
                state_dict = torch.load(bin_path, **kwargs)
        else:
            raise FileNotFoundError(f"No model weights found in {model_path}")

    # 2. Check for packed weights
    has_packed_weights = any(key.endswith("_packed") for key in state_dict.keys())

    if not has_packed_weights:
        print("INFO: Legacy/Unpacked format detected.")
        return state_dict, False

    print("INFO: Packed format detected. Unpacking...")
    packed_components = defaultdict(dict)
    final_state_dict = {}

    # Regex to group: (layer.path).(param)_(packed|shape)
    # Example: model.layers.0.self_attn.q_proj.U_packed -> prefix=...q_proj, param=U, suffix=packed
    pattern = re.compile(r"^(.*)\.([^.]+?)_(packed|shape)$")

    for key, value in state_dict.items():
        match = pattern.match(key)
        if match:
            prefix, param_name, suffix_type = match.groups()
            packed_components[prefix][f"{param_name}_{suffix_type}"] = value
        else:
            final_state_dict[key] = value

    # Unpack
    for prefix, components in packed_components.items():
        # Identify parameter names (e.g., U, V, U_R)
        param_names = {k.rsplit('_', 1)[0] for k in components.keys() if k.endswith("_packed")}

        for name in param_names:
            packed_tensor = components.get(f"{name}_packed")
            shape_tensor = components.get(f"{name}_shape")

            if packed_tensor is not None and shape_tensor is not None:
                shape = tuple(shape_tensor.tolist())
                # Unpacking typically runs faster on CUDA if available, but staying on CPU to avoid OOM
                # If speed is issue, move packed_tensor to cuda, unpack, move back.
                from quantization.utils.binary_packer import binary_unpacker
                unpacked = binary_unpacker(packed_tensor, shape).to(torch_dtype)
                final_state_dict[f"{prefix}.{name}"] = unpacked
            else:
                # Should not happen if save was correct
                pass

    return final_state_dict, True


def load_quantized_model(model_path: str, quant_args, torch_dtype, device: str = "auto"):
    """
    Loads a pre-quantized model from a directory.
    Replaces `modeling.py` usage with dynamic `AutoModel` patching.
    
    Args:
        model_path: Path to the quantized model directory
        quant_args: Namespace containing quantization parameters (quant_func, eff_bit, etc.)
        torch_dtype: Data type for model parameters
        device: Target device (default: "auto")
    """
    if not os.path.isdir(model_path):
        raise ValueError(f"Model path must be a directory: {model_path}")

    print(f"INFO: Loading configuration from '{model_path}'...")
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)

    # Automatically add quantization parameters to config
    quant_params = {
        "quant_func": getattr(quant_args, "quant_func", "STEBinary"),
        "eff_bit": getattr(quant_args, "eff_bit", 1.0),
        "split_dim": getattr(quant_args, "split_dim", 1024),
        "residual": getattr(quant_args, "residual", False),
        "is_po2": getattr(quant_args, "is_po2", False),
        "num_expert": getattr(quant_args, "num_expert", 4),
        "kv_factor": getattr(quant_args, "kv_factor", 1.0),
        "min_split_dim": getattr(quant_args, "min_split_dim", 8),
        "use_itq": getattr(quant_args, "use_itq", False),
        "itq_n_iter": getattr(quant_args, "itq_n_iter", 50),
    }
    for key, value in quant_params.items():
        if not hasattr(config, key):
            setattr(config, key, value)

    # 1. Create Skeleton Model on Meta Device
    print(f"INFO: Initializing structure for {config.model_type} on meta...")
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)

    model.to_empty(device="cpu")

    print("INFO: Cleaning uninitialized memory and fixing buffers...")
    for param in model.parameters():
        param.data.zero_()

    for name, module in model.named_modules():
        if "rotary_emb" in name.lower() or "rope" in name.lower():
            if hasattr(module, 'inv_freq'):
                module.__init__(config=config) if 'config' in module.__init__.__code__.co_varnames else module.__init__(
                    module.dim, module.max_position_embeddings, module.base)

    # 2. Patch Model Structure
    if not hasattr(quant_args, "model_id") or not quant_args.model_id:
        quant_args.model_id = model_path

    print("INFO: Applying quantization patch...")
    model = apply_littlebit_patch(model, quant_args, do_train=False)

    # 3. Load Weights
    print("INFO: Loading state dictionary...")
    state_dict, was_unpacked = _load_and_process_state_dict(model_path, torch_dtype)

    # Load into model
    missing, unexpected = model.load_state_dict(state_dict, strict=False, assign=True)
    if missing:
        print(f"WARNING: Missing keys: {missing[:10]}...")
    if unexpected:
        print(f"WARNING: Unexpected keys: {unexpected[:10]}...")

    print("INFO: Tying model weights...")
    try:
        model.tie_weights()
    except Exception as e:
        print(f"WARN: tie_weights() failed: {e}")

    if hasattr(model, "lm_head") and getattr(getattr(model.lm_head, "weight", None), "is_meta", False):
        print("WARN: lm_head.weight is still on meta. Manually resolving...")
        if hasattr(model, "get_input_embeddings"):
            emb = model.get_input_embeddings()
            if emb is not None and hasattr(emb, "weight") and not getattr(emb.weight, "is_meta", False):
                model.lm_head.weight = emb.weight
        else:
            w = model.lm_head.weight
            model.lm_head.weight = nn.Parameter(torch.empty(w.shape, device="cpu", dtype=torch_dtype))
            model.lm_head.weight.data.zero_()

    del state_dict
    gc.collect()

    # 4. Post-Process (Binarization Flag & casting)
    if was_unpacked:
        print("INFO: Setting modules to binarized inference mode.")
        for module in model.modules():
            if isinstance(module, LittleBitLinear):
                module._binarized = True
    else:
        print(f"INFO: Legacy format. Casting params to {torch_dtype}...")
        for param in model.parameters():
            if param.dtype != torch_dtype:
                param.data = param.data.to(torch_dtype)

    # Ensure head is correct dtype
    if hasattr(model, 'lm_head') and model.lm_head is not None:
        model.lm_head.to(torch_dtype)

    print("INFO: Checking for remaining meta tensors...")
    for name, module in model.named_modules():
        # Handle Parameters
        for param_name, param in list(module.named_parameters(recurse=False)):
            if param.device.type == 'meta':
                # print(f"WARN: Materializing meta parameter '{name}.{param_name}' with zeros.")
                # Create new parameter on CPU and replace
                new_param = nn.Parameter(torch.zeros_like(param, device="cpu", dtype=torch_dtype),
                                         requires_grad=param.requires_grad)
                setattr(module, param_name, new_param)

    # 5. Move to device
    if device == "auto":
        target_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        target_device = torch.device(device)

    print(f"INFO: Moving model to {target_device}...")
    model.to(target_device)

    print(f"Model ready on {target_device}")
    return model
