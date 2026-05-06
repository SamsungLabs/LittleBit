"""
Hugging Face Hub integration for LittleBit models.

Provides PyTorchModelHubMixin-based classes for native `from_pretrained()` and `push_to_hub()` support.

Usage:
    # Direct usage
    from quantization.hub import LittleBitModel
    model = LittleBitModel.from_pretrained("username/littlebit-llama-7b-0.1bpw")
    model.push_to_hub("username/my-littlebit-model")

    # Via AutoModelForCausalLM (requires proper config.json with auto_map)
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained("username/littlebit-llama-7b-0.1bpw")
"""

import os
import json
import gc
import torch
import tempfile
import shutil
import argparse
import torch.nn as nn
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import dataclass, asdict
from huggingface_hub import PyTorchModelHubMixin, snapshot_download, create_repo
from transformers import AutoConfig, AutoModelForCausalLM

from .utils.quant_util import load_quantized_model, apply_littlebit_patch
from .modules import LittleBitLinear


@dataclass
class LittleBitConfig:
    """Quantization configuration for LittleBit models."""
    quant_func: str = "STEBinary"
    eff_bit: float = 1.0
    split_dim: int = 1024
    residual: bool = False
    kv_factor: float = 1.0
    min_split_dim: int = 8
    use_itq: bool = False
    itq_n_iter: int = 50

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "LittleBitConfig":
        valid_keys = set(cls.__dataclass_fields__.keys())
        return cls(**{k: v for k, v in data.items() if k in valid_keys})


class LittleBitModel(nn.Module, PyTorchModelHubMixin):
    """
    LittleBit model wrapper with native HuggingFace Hub support.
    
    This class wraps a quantized model and provides native `from_pretrained()` and 
    `push_to_hub()` methods. It also supports `AutoModelForCausalLM.from_pretrained()`
    when the model has proper `auto_map` configuration in `config.json`.

    Usage:
        >>> from quantization.hub import LittleBitModel
        >>> 
        >>> # Load from Hub
        >>> model = LittleBitModel.from_pretrained(
        ...     "username/littlebit-llama-7b-0.1bpw",
        ...     torch_dtype=torch.bfloat16
        ... )
        >>> 
        >>> # Or via AutoModel (requires auto_map in config.json)
        >>> from transformers import AutoModelForCausalLM
        >>> model = AutoModelForCausalLM.from_pretrained(
        ...     "username/littlebit-llama-7b-0.1bpw"
        ... )
        >>> 
        >>> # Push to Hub
        >>> model.push_to_hub("username/my-littlebit-model")
    """
    def __init__(
        self,
        model: torch.nn.Module,
        config: LittleBitConfig,
        base_model_id: Optional[str] = None,
    ):
        """
        Initialize the LittleBit wrapper.
        
        Args:
            model: The underlying quantized PyTorch model
            config: LittleBit quantization configuration
            base_model_id: Original model ID (e.g., "meta-llama/Llama-2-7b")
        """
        super().__init__()
        self.model = model
        self._littlebit_config = config
        self._base_model_id = base_model_id

    @property
    def config(self):
        """Return the underlying model's config."""
        return self.model.config

    @property
    def littlebit_config(self) -> LittleBitConfig:
        """Return the LittleBit quantization config."""
        return self._littlebit_config

    def _save_pretrained(self, save_directory: Path, **kwargs):
        """
        Save the model to a local directory.
        
        Saves:
            - littlebit_config.json: Quantization parameters
            - config.json: Model config with auto_map for AutoModel support
            - model.safetensors or model_state.pt: Model weights
            - tokenizer files (if present)
        """
        save_directory = Path(save_directory)
        save_directory.mkdir(parents=True, exist_ok=True)

        # Save LittleBit quantization config
        quant_config_path = save_directory / "littlebit_config.json"
        with open(quant_config_path, "w") as f:
            json.dump(self._littlebit_config.to_dict(), f, indent=2)

        # Save base model info
        if self._base_model_id:
            base_config_path = save_directory / "base_model.json"
            with open(base_config_path, "w") as f:
                json.dump({"model_id": self._base_model_id}, f, indent=2)

        # Update and save model config with auto_map
        model_config = dict(self.model.config)
        model_config["auto_map"] = {"AutoModelForCausalLM": "quantization.hub.LittleBitModel"}

        # Add quantization params to config for auto-detection
        model_config["quant_func"] = self._littlebit_config.quant_func
        model_config["eff_bit"] = self._littlebit_config.eff_bit
        model_config["split_dim"] = self._littlebit_config.split_dim
        model_config["residual"] = self._littlebit_config.residual
        model_config["use_itq"] = self._littlebit_config.use_itq
        model_config["itq_n_iter"] = self._littlebit_config.itq_n_iter

        config_path = save_directory / "config.json"
        with open(config_path, "w") as f:
            json.dump(model_config, f, indent=2)

        # Save weights as safetensors (preferred format)
        try:
            from safetensors.torch import save_file
            state_dict = self.model.state_dict()
            save_file(state_dict, str(save_directory / "model.safetensors"))
        except ImportError:
            # Fallback to torch.save
            state_dict = self.model.state_dict()
            torch.save(state_dict, save_directory / "model_state.pt")

        # Create/Update sharded index if needed
        index_file = save_directory / "model.safetensors.index.json"
        if not index_file.exists():
            # Create a minimal index file for single-shard model
            weight_map = {key: "model.safetensors" for key in self.model.state_dict().keys()}
            with open(index_file, "w") as f:
                json.dump(
                    {
                        "metadata": {
                            "total_size": sum(p.numel() * p.element_size() for p in self.model.parameters())
                        },
                        "weight_map": weight_map
                    }, f, indent=2)

        # Copy tokenizer files if they exist
        tokenizer_files = ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "vocab.json"]
        for tf in tokenizer_files:
            if os.path.exists(tf):
                shutil.copy(tf, save_directory / tf)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        torch_dtype: torch.dtype = torch.bfloat16,
        device: str = "auto",
        **kwargs,
    ) -> "LittleBitModel":
        """
        Load model from local path or HuggingFace Hub.
        """
        # 1. Separate quantization-related parameters from kwargs
        # (to prevent conflicts with HuggingFace Hub download logic)
        littlebit_kwargs = {k: v for k, v in kwargs.items() if k in LittleBitConfig.__dataclass_fields__}
        for k in list(littlebit_kwargs.keys()):
            kwargs.pop(k)

        # Download from Hub if this looks like a Hub model ID
        local_path = pretrained_model_name_or_path
        if not os.path.isdir(pretrained_model_name_or_path):
            try:
                # Use snapshot_download to pull the entire repository securely into cache
                local_dir = snapshot_download(
                    repo_id=pretrained_model_name_or_path,
                    cache_dir=kwargs.get("cache_dir"),
                    token=kwargs.get("token"),
                )
                local_path = local_dir
            except Exception as e:
                raise ValueError(f"Failed to download model from HuggingFace Hub: {e}")

        # 2. First, attempt to load littlebit_config.json (for newer models)
        quant_config_path = Path(local_path) / "littlebit_config.json"
        if quant_config_path.exists():
            with open(quant_config_path) as f:
                config_dict = json.load(f)
            config = LittleBitConfig.from_dict(config_dict)
        else:
            config = LittleBitConfig()

        # 3. Attempt to load from config.json (Fallback for older models)
        model_config_path = Path(local_path) / "config.json"
        if model_config_path.exists():
            with open(model_config_path) as f:
                model_config = json.load(f)
            for key in [
                    "quant_func", "eff_bit", "split_dim", "residual", "quant_mod", "num_expert", "kv_factor", "use_itq",
                    "itq_n_iter"
            ]:
                # Apply fallback if the key exists in model_config and hasn't been explicitly set in config
                if key in model_config and (not hasattr(config, key)
                                            or getattr(config, key) == getattr(LittleBitConfig(), key)):
                    setattr(config, key, model_config[key])

        # 4. Final overwrite with kwargs provided directly by the user (highest priority)
        for k, v in littlebit_kwargs.items():
            setattr(config, k, v)

        # Create quant_args namespace
        quant_args = argparse.Namespace(
            quant_func=getattr(config, "quant_func", "STEBinary"),
            eff_bit=getattr(config, "eff_bit", 1.0),
            split_dim=getattr(config, "split_dim", 1024),
            residual=getattr(config, "residual", False),
            kv_factor=getattr(config, "kv_factor", 1.0),
            min_split_dim=getattr(config, "min_split_dim", 8),
            quant_mod=getattr(config, "quant_mod", "LittleBitLinear"),
            use_itq=getattr(config, "use_itq", False),
            itq_n_iter=getattr(config, "itq_n_iter", 50),
            model_id=local_path,
        )

        # Load model using existing infrastructure
        model = load_quantized_model(
            model_path=local_path,
            quant_args=quant_args,
            torch_dtype=torch_dtype,
            device=device,
        )

        # Load base model info (Optional)
        base_model_id = None
        base_config_path = Path(local_path) / "base_model.json"
        if base_config_path.exists():
            with open(base_config_path) as f:
                base_info = json.load(f)
            base_model_id = base_info.get("model_id")

        return cls(model, config, base_model_id=base_model_id)

    def _generate_readme(self, repo_id: str) -> str:
        """Generate README.md content with model metadata."""
        model_name = repo_id.split("/")[-1]

        sections = [
            # YAML frontmatter
            "---\n"
            "license: other\n"
            "tags:\n"
            "- quantization\n"
            "- littlebit\n"
            f"- {self.config.model_type}\n"
            "---\n",

            # Title
            f"# {model_name}\n\n"
            "A LittleBit quantized model.\n",

            # Quantization config
            "## Quantization Config\n\n" + "\n".join([
                f"- **Quantization Function**: {self._littlebit_config.quant_func}",
                f"- **Effective Bits**: {self._littlebit_config.eff_bit}",
                f"- **Split Dim**: {self._littlebit_config.split_dim}",
                f"- **Residual**: {self._littlebit_config.residual}",
                f"- **KV Factor**: {self._littlebit_config.kv_factor}",
            ]) + "\n",

            # Usage
            "## Usage\n\n"
            "Load the model using `transformers.AutoModelForCausalLM`:\n\n"
            f"```python\n"
            f"from transformers import AutoModelForCausalLM\n\n"
            f"model = AutoModelForCausalLM.from_pretrained(\"{repo_id}\")\n"
            f"```\n\n"
            "Or with explicit `LittleBitModel`:\n\n"
            f"```python\n"
            f"from quantization import LittleBitModel\n\n"
            f"model = LittleBitModel.from_pretrained(\"{repo_id}\")\n"
            f"```\n",

            # Original model
            "## Original Model\n\n" +
            (f"Base model: [{self._base_model_id}](https://huggingface.co/{self._base_model_id})\n"
             if self._base_model_id else "- Not specified\n"),

            # Model details
            "## Model Details\n\n" + "\n".join([
                f"- **Model Type**: {self.config.model_type}",
                f"- **Hidden Size**: {getattr(self.config, 'hidden_size', 'N/A')}",
                f"- **Num Attention Heads**: {getattr(self.config, 'num_attention_heads', 'N/A')}",
                f"- **Num Hidden Layers**: {getattr(self.config, 'num_hidden_layers', 'N/A')}",
            ]),

            # Citation
            "## Citation\n\n"
            "If you use this model, please cite:\n\n"
            "```\n"
            "@inproceedings{littlebit2025,\n"
            "  title={LittleBit: Ultra Low-Bit Quantization via Latent Factorization},\n"
            "  author={Lee, Banseok and Kim, Dongkyu and You, Youngcheon and Kim, Youngmin},\n"
            "  booktitle={NeurIPS},\n"
            "  year={2025}\n"
            "}\n"
            "```",
        ]

        return "\n\n".join(sections)

    def push_to_hub(
        self,
        repo_id: str,
        use_temp_dir: bool = True,
        commit_message: str = "Push LittleBit quantized model to Hub",
        private: bool = False,
        token: Optional[str] = None,
        local_dir: Optional[str] = None,
        **kwargs,
    ) -> str:
        """
        Push the model to HuggingFace Hub.
        
        Automatically saves quantization config and model weights, then pushes
        to the Hub with appropriate metadata.

        Args:
            repo_id: Repository ID (e.g., "username/my-littlebit-model")
            use_temp_dir: Use temporary directory for preparation
            commit_message: Git commit message for the push
            private: Create repository as private
            token: HuggingFace API token (uses cached token if None)
            local_dir: Optional alternative directory for local save before push

        Returns:
            URL of the pushed repository

        Example:
            >>> model.push_to_hub("my-littlebit-7b-0.1bpw", private=True)
        """
        # Create repo (may error if already exists, that's fine)
        try:
            create_repo(repo_id, exist_ok=True, private=private, token=token)
        except Exception:
            pass  # Repo may already exist

        # Prepare save directory
        if local_dir:
            save_dir = Path(local_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
        else:
            save_dir = Path(tempfile.mkdtemp())

        try:
            # Save all files
            self._save_pretrained(save_dir)

            # Add README with model metadata
            readme_content = self._generate_readme(repo_id)

            with open(save_dir / "README.md", "w") as f:
                f.write(readme_content)

            # Upload to Hub
            api_url = super().push_to_hub(
                repo_id=repo_id,
                commit_message=commit_message,
                token=token,
                use_temp_dir=use_temp_dir,
                **kwargs,
            )
            return api_url

        finally:
            # Cleanup temp directory if we created it
            if not local_dir and save_dir.exists():
                shutil.rmtree(save_dir, ignore_errors=True)

    def to(self, *args, **kwargs):
        """Delegate to() call to underlying model."""
        self.model = self.model.to(*args, **kwargs)
        return self

    def cuda(self, **kwargs):
        """Delegate cuda() call to underlying model."""
        self.model = self.model.cuda(**kwargs)
        return self

    def cpu(self):
        """Delegate cpu() call to underlying model."""
        self.model = self.model.cpu()
        return self

    def state_dict(self, *args, **kwargs):
        """Get state dict from underlying model."""
        return self.model.state_dict(*args, **kwargs)

    def load_state_dict(self, state_dict, strict=True, assign=False):
        """Load state dict into underlying model."""
        return self.model.load_state_dict(state_dict, strict=strict, assign=assign)

    def parameters(self, recurse=True):
        """Iterate over model parameters."""
        return self.model.parameters()

    def named_parameters(self, recurse=True):
        """Iterate over named model parameters."""
        return self.model.named_parameters(recurse)

    def modules(self):
        """Iterate over submodules."""
        return self.model.modules()

    def children(self):
        """Iterate over child modules."""
        return self.model.children()

    def __getattr__(self, name: str):
        """Delegate attribute access to underlying model.
        
        This allows the wrapper to be used transparently with
        model methods like .generate(), .forward(), etc.
        """
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)

    def forward(self, *args, **kwargs):
        """Delegate forward pass to underlying model."""
        return self.model.forward(*args, **kwargs)

    def generate(self, *args, **kwargs):
        """Delegate generate() to underlying model."""
        return self.model.generate(*args, **kwargs)

    def forward(self, *args, **kwargs):
        """Delegate forward pass to underlying model."""
        return self.model.forward(*args, **kwargs)
