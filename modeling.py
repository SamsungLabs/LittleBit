import torch.nn as nn
from transformers.models.gemma2 import Gemma2ForCausalLM, Gemma2Model
from transformers.models.gemma3 import Gemma3ForCausalLM, Gemma3TextConfig, Gemma3TextModel
from transformers.models.llama import LlamaForCausalLM, LlamaModel
from transformers.models.opt import OPTForCausalLM, OPTModel
from transformers.models.phi3 import Phi3ForCausalLM as Phi4ForCausalLM
from transformers.models.phi3 import Phi3Model as Phi4Model
from transformers.models.qwen2 import Qwen2ForCausalLM as QwQForCausalLM
from transformers.models.qwen2 import Qwen2Model as QwQModel


class LittleBitForCausalLMBase:
    @classmethod
    def from_pretrained(cls, model_id, *model_args, **kwargs):
        cls.extra_config = kwargs.pop("extra_config", None)
        return super().from_pretrained(model_id, *model_args, **kwargs)

    def _common_init(self, config, base_model_class):
        self.model = base_model_class(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.post_init()

        import re
        from quantization.utils import get_quant_func_and_mod, patch_inst

        args = self.extra_config
        quant_func, quant_mod = get_quant_func_and_mod(args.quant_func, args.quant_mod)

        mapping = {nn.Linear: quant_mod}

        KV_PATTERN = [re.compile(r'\.k_proj$'), re.compile(r'\.v_proj$')]
        convert_kwargs = [
            ([nn.Linear], {
                "do_train": False,
                "quant_func": quant_func,
                "residual": args.residual,
                "split_dim": args.split_dim,
                "eff_bit": args.eff_bit,
            }),
            (KV_PATTERN, {
                "ratio_factor": args.kv_factor,
            }),
        ]

        if "phi" in args.model_id.lower():
            from transformers.models.phi3.modeling_phi3 import Phi3Attention
            from quantization.modules.attention import PhiQKVSplitAttention

            mapping.update({Phi3Attention: PhiQKVSplitAttention})
            convert_kwargs.append(([Phi3Attention], {'config': self.model.config}))

        patch_inst(
            self.model,
            convert_kwargs=convert_kwargs,
            mapping=mapping,
            exclude_names=["lm_head"],
        )


class LittleBitLlamaForCausalLM(LittleBitForCausalLMBase, LlamaForCausalLM):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self._common_init(config, LlamaModel)


class LittleBitPhi4ForCausalLM(LittleBitForCausalLMBase, Phi4ForCausalLM):
    _tied_weights_keys = ["lm_head.weight"]
    _tp_plan = {"lm_head": "colwise_rep"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}

    def __init__(self, config):
        super().__init__(config)
        self._common_init(config, Phi4Model)


class LittleBitGemma2ForCausalLM(LittleBitForCausalLMBase, Gemma2ForCausalLM):
    _tied_weights_keys = ["lm_head.weight"]
    _tp_plan = {"lm_head": "colwise_rep"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}

    def __init__(self, config):
        super().__init__(config)
        self._common_init(config, Gemma2Model)


class LittleBitGemma3ForCausalLM(LittleBitForCausalLMBase, Gemma3ForCausalLM):
    _tied_weights_keys = ["lm_head.weight"]
    _tp_plan = {"lm_head": "colwise_rep"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}
    config_class = Gemma3TextConfig
    base_model_prefix = "language_model"

    def __init__(self, config: Gemma3TextConfig):
        super().__init__(config)
        self._common_init(config, Gemma3TextModel)


class LittleBitQwQForCausalLM(LittleBitForCausalLMBase, QwQForCausalLM):
    _tied_weights_keys = ["lm_head.weight"]
    _tp_plan = {"lm_head": "colwise_rep"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}

    def __init__(self, config):
        super().__init__(config)
        self._common_init(config, QwQModel)


class LittleBitOPTForCausalLM(LittleBitForCausalLMBase, OPTForCausalLM):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        config.hidden_size = config.word_embed_proj_dim
        self._common_init(config, OPTModel)
