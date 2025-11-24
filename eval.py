import argparse
import torch
import torch.nn as nn
from lm_eval import evaluator
from lm_eval.base import BaseLM
from tqdm import tqdm
from transformers import AutoTokenizer

from modeling import (LittleBitGemma2ForCausalLM, LittleBitGemma3ForCausalLM, LittleBitLlamaForCausalLM,
                      LittleBitOPTForCausalLM, LittleBitPhi4ForCausalLM, LittleBitQwQForCausalLM)
from utils.datautils import get_eval_loaders


def str2bool(value):
    if value.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif value.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise Exception(f'Boolean value expected: {value}')


class EvalLM(BaseLM):
    def __init__(self, model, tokenizer, batch_size=1, accelerator=None):
        super().__init__()
        self.batch_size_per_gpu = batch_size
        self.seqlen = 2048
        self.tokenizer = tokenizer
        if accelerator is not None:
            self.accelerator = accelerator
            self._device = accelerator.device
            self.model = model
        else:
            self.accelerator = None
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model = model.to(self._device)

        self.model.eval()
        self.vocab_size = self.tokenizer.vocab_size

    @property
    def eot_token_id(self):
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        actual_model = self.model.module if hasattr(self.model, "module") else self.model
        return getattr(actual_model.config, "n_ctx", actual_model.config.max_position_embeddings)

    @property
    def max_gen_toks(self):
        return 256

    @property
    def batch_size(self):
        return self.batch_size_per_gpu

    def tok_encode(self, string: str):
        return self.tokenizer.encode(string, add_special_tokens=False)

    def tok_decode(self, tokens):
        return self.tokenizer.decode(tokens)

    def _model_call(self, inps):
        with torch.no_grad():
            outputs = self.model(inps, use_cache=False)
            if hasattr(outputs, "logits"):
                return outputs.logits
            elif isinstance(outputs, dict) and "logits" in outputs:
                return outputs["logits"]
            else:
                return outputs

    def _model_generate(self, context, max_length, eos_token_id):
        with torch.no_grad():
            actual_model = self.model.module if hasattr(self.model, "module") else self.model
            return actual_model.generate(
                context,
                max_length=max_length,
                eos_token_id=eos_token_id,
                do_sample=False,
            )

    @property
    def device(self):
        # TODO: fix multi-gpu
        return self._device


@torch.no_grad()
def evaluate_model(
    model,
    tokenizer,
    tasks,
    eval_ppl="",
    num_fewshot=0,
    limit=-1,
    batch_size=1,
    accelerator=None,
):
    lm = EvalLM(model, tokenizer, batch_size=batch_size, accelerator=accelerator)
    results = {}
    if eval_ppl:
        datasets = eval_ppl.split(",")
        for dataset in datasets:
            msg = f"[INFO] Starting PPL eval for: {dataset}"
            if accelerator is not None:
                accelerator.print(msg)
            else:
                print(msg)

            testloader = get_eval_loaders(dataset, tokenizer)
            testenc = testloader.input_ids
            nsamples = testenc.numel() // lm.seqlen

            lm.model.eval()
            nlls = []
            for i in tqdm(range(nsamples), disable=(accelerator is None
                                                    or not getattr(accelerator, "is_local_main_process", True))):
                batch = testenc[:, (i * lm.seqlen):(i + 1) * lm.seqlen].to(lm.device, dtype=torch.long)
                outputs = lm.model(batch, use_cache=False)
                logits = outputs.logits if hasattr(outputs, "logits") else outputs["logits"]

                shift_logits = logits[:, :-1, :]
                shift_labels = batch[:, 1:]

                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.reshape(-1))
                neg_log_likelihood = loss.float() * lm.seqlen
                nlls.append(neg_log_likelihood)

            ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * lm.seqlen))
            out_msg = f"[{dataset}] PPL = {ppl.item()}"
            if accelerator is not None:
                accelerator.print(out_msg)
            else:
                print(out_msg)

            results[dataset] = ppl.item()

    if tasks:
        harness_results = evaluator.simple_evaluate(
            model=lm,
            tasks=tasks.split(","),
            batch_size=batch_size,
            num_fewshot=num_fewshot,
            limit=None if limit == -1 else limit,
            no_cache=True,
        )
        harness_results = harness_results["results"]
        results.update(harness_results)
        msg = f"Zero-shot tasks results: {harness_results}"
        if accelerator is not None:
            accelerator.print(msg)
        else:
            print(msg)

    return results


def main(args):
    lm_dict = {
        "llama": LittleBitLlamaForCausalLM,
        "gemma2": LittleBitGemma2ForCausalLM,
        "gemma3": LittleBitGemma3ForCausalLM,
        "phi4": LittleBitPhi4ForCausalLM,
        "qwq": LittleBitQwQForCausalLM,
        "opt": LittleBitOPTForCausalLM,
    }

    if args.model_type not in lm_dict:
        raise KeyError(f"Invalid model type: {args.model_type}. Available model types: {list(lm_dict.keys())}")

    LM = lm_dict[args.model_type]

    if args.use_accelerator:
        from accelerate import Accelerator
        from accelerate.utils import DeepSpeedPlugin

        deepspeed_plugin = DeepSpeedPlugin(zero_stage=3, offload_optimizer_device="cpu", offload_param_device="cpu")
        accelerator = Accelerator(deepspeed_plugin=deepspeed_plugin, mixed_precision="bf16")
        model = LM.from_pretrained(
            args.model_id,
            device_map=None,
            torch_dtype=torch.bfloat16,
            extra_config=args,
        )
        tokenizer = AutoTokenizer.from_pretrained(args.model_id, use_fast=False, trust_remote_code=True)

        from torch.utils.data import TensorDataset, DataLoader
        dummy_data = torch.zeros((8, 1), dtype=torch.float32)
        dummy_dataset = TensorDataset(dummy_data)
        dummy_loader = DataLoader(dummy_dataset, batch_size=1)
        model, dummy_loader = accelerator.prepare(model, dummy_loader)
    else:
        accelerator = None
        model = LM.from_pretrained(
            args.model_id,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            extra_config=args,
        )
        tokenizer = AutoTokenizer.from_pretrained(args.model_id, device_map="auto", use_fast=False,
                                                  trust_remote_code=True)

    _ = evaluate_model(
        model=model,
        tokenizer=tokenizer,
        tasks=args.zeroshot_task,
        eval_ppl=args.ppl_task,
        batch_size=args.batch_size,
        accelerator=accelerator,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model Evaluation Script with optional Accelerator/DeepSpeed support")
    parser.add_argument("--local_rank", type=int, default=-1, help="(Accelerator/DeepSpeed related) local rank")
    parser.add_argument("--use_accelerator", type=str2bool, default=False, help="Whether to use Accelerator/DeepSpeed")
    parser.add_argument("--model_type", type=str, default=None,
                        help="Model type (llama, gemma2, gemma3, phi4, qwq, opt)")
    parser.add_argument("--model_id", type=str, default="meta-llama/Llama-2-7b-hf", help="Model ID or path")
    parser.add_argument("--ppl_task", type=str, default="wikitext2,c4",
                        help="Perplexity evaluation dataset (comma-separated)")
    parser.add_argument("--zeroshot_task", type=str,
                        default="boolq,piqa,hellaswag,winogrande,arc_easy,arc_challenge,openbookqa",
                        help="Zero-shot evaluation tasks (comma-separated)")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")

    parser.add_argument("--quant_func", type=str, default="STEBinary")
    parser.add_argument("--quant_mod", type=str, default="LittleBitLinear")

    parser.add_argument("--residual", type=str2bool, default=False)
    parser.add_argument("--split_dim", type=int, default=1024)
    parser.add_argument("--eff_bit", type=float, default=1.0)
    parser.add_argument("--kv_factor", type=float, default=1.0)

    args = parser.parse_args()

    main(args)
