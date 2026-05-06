import os
import argparse
import json
import torch
import torch._dynamo
import torch.nn as nn
from lm_eval import evaluator
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

from quantization.hub import LittleBitModel
from utils.datautils import get_eval_loaders, load_tokenizer


def str2bool(value):
    if value.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif value.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError(f'Boolean value expected: {value}')


@torch.no_grad()
def evaluate_model(
    model,
    tokenizer,
    tasks_str,
    eval_ppl="",
    num_fewshot=0,
    limit=-1,
    batch_size=1,
    accelerator=None,
    seqlen=None,
):
    results = {}

    # --- PPL Evaluation ---
    if eval_ppl:
        datasets = eval_ppl.split(",")
        for dataset in datasets:
            msg = f"[INFO] Starting PPL eval for: {dataset}"
            if accelerator: accelerator.print(msg)
            else: print(msg)

            try:
                testloader = get_eval_loaders(dataset, tokenizer)
                testenc = testloader.input_ids
            except Exception as e:
                print(f"Failed to load PPL dataset {dataset}: {e}")
                continue

            actual_model = model.module if hasattr(model, "module") else model
            # Handle config naming variations (n_ctx, max_position_embeddings, etc.)
            # If seqlen is None, try to get it from config, otherwise use provided value
            if seqlen is None:
                seqlen = getattr(actual_model.config, "n_ctx",
                                 getattr(actual_model.config, "max_position_embeddings", 2048))

            nsamples = testenc.numel() // seqlen
            if nsamples == 0:
                print(f"Not enough data for PPL evaluation on {dataset} with seqlen {seqlen}. Skipping.")
                continue

            model.eval()
            nlls = []
            device = accelerator.device if accelerator else next(model.parameters()).device

            # Loop over chunks
            for i in tqdm(range(nsamples), disable=not (accelerator is None or accelerator.is_local_main_process)):
                batch = testenc[:, (i * seqlen):((i + 1) * seqlen)].to(device)
                outputs = model(batch, use_cache=False)
                logits = outputs.logits if hasattr(outputs, "logits") else outputs["logits"]

                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = batch[:, 1:].contiguous()

                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                neg_log_likelihood = loss.float() * (seqlen - 1)
                nlls.append(neg_log_likelihood)

            ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * (seqlen - 1)))
            out_msg = f"[{dataset}] PPL = {ppl.item():.4f}"
            if accelerator: accelerator.print(out_msg)
            else: print(out_msg)
            results[dataset] = {"ppl": ppl.item()}

    # --- Harness Evaluation ---
    if tasks_str:
        task_names = [task.strip() for task in tasks_str.split(',') if task.strip()]
        if task_names:
            if accelerator:
                current_device = None
            else:
                raw_device = next(model.parameters()).device
                current_device = str(raw_device)

            harness_results = evaluator.simple_evaluate(
                model="hf",
                model_args={
                    "pretrained": model,
                    "tokenizer": tokenizer,
                    "max_length": seqlen,
                    **({
                        "accelerator": accelerator
                    } if accelerator else {})
                },
                tasks=task_names,
                num_fewshot=num_fewshot,
                batch_size=batch_size,
                limit=None if limit == -1 else limit,
                device=current_device,
            )
            results.update(harness_results["results"])

            msg = f"Zero-shot tasks results: {json.dumps(harness_results['results'], indent=2)}"
            if accelerator: accelerator.print(msg)
            else: print(msg)

    return results


def main(args):
    torch_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32

    model = None

    # =========================================================================
    # Strategy 1: Pure FP Evaluation (Baseline)
    # =========================================================================
    if args.fp_eval:
        print(f"INFO: Loading original FP model '{args.model_id}' for baseline evaluation.")
        model = AutoModelForCausalLM.from_pretrained(args.model_id, low_cpu_mem_usage=True, torch_dtype=torch_dtype,
                                                     trust_remote_code=True)
        print("FP model loaded.")

    # =========================================================================
    # Strategy 2: Load Pre-Quantized Model via LittleBitModel (Local or Hub)
    # =========================================================================
    else:
        print(f"INFO: Loading LittleBit model from '{args.model_id}'.")

        # Extract explicit quantization args provided via CLI
        # Only collect arguments that are NOT None (i.e., user explicitly set them)
        quant_kwargs = {}
        quant_keys = [
            "quant_func", "quant_mod", "residual", "split_dim", "eff_bit", "kv_factor", "use_itq", "itq_n_iter"
        ]
        for key in quant_keys:
            val = getattr(args, key)
            if val is not None:
                quant_kwargs[key] = val
                print(f"INFO: Overriding config with CLI argument: {key}={val}")

        # LittleBitModel.from_pretrained reads config files by default,
        # but will prioritize the explicit `quant_kwargs` if provided.
        model = LittleBitModel.from_pretrained(args.model_id, torch_dtype=torch_dtype, device="auto", **quant_kwargs)
        print("LittleBit model loaded successfully.")

    if model is None:
        raise RuntimeError("Model could not be loaded based on the provided arguments.")

    # --- Tokenizer Setup ---
    tokenizer = load_tokenizer(args.model_id)

    # --- Accelerator / Device Setup ---
    if args.use_accelerator:
        from accelerate import Accelerator
        # mixed_precision='no' usually better for eval if we explicitly handled dtypes,
        # but 'bf16' is fine if supported.
        accelerator = Accelerator(mixed_precision="bf16" if torch_dtype == torch.bfloat16 else "no")

        # Dummy prep for accelerator to handle device placement
        from torch.utils.data import TensorDataset, DataLoader
        dummy_loader = DataLoader(TensorDataset(torch.zeros((1, 1))))
        model, dummy_loader = accelerator.prepare(model, dummy_loader)
    else:
        accelerator = None
        # If not already on GPU (from_pretrained might have put it there via device="auto"), move it.
        device = next(model.parameters()).device
        if device.type == 'cpu' and torch.cuda.is_available():
            target_device = torch.device("cuda")
            model.to(target_device)
            print(f"Model moved to {target_device}")

    # --- Run Evaluation ---
    _ = evaluate_model(
        model=model,
        tokenizer=tokenizer,
        tasks_str=args.zeroshot_task,
        eval_ppl=args.ppl_task,
        num_fewshot=args.num_fewshot,
        limit=args.limit,
        batch_size=args.batch_size,
        accelerator=accelerator,
        seqlen=args.seqlen,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unified Model Evaluation Script")
    parser.add_argument("--use_accelerator", type=str2bool, default=False)

    parser.add_argument("--model_id", type=str, default="meta-llama/Llama-2-7b-hf")
    parser.add_argument("--fp_eval", type=str2bool, default=False)

    # Note: Default values are set to None.
    # This allows `LittleBitModel` to load parameters from config files natively,
    # while still allowing users to override specific parameters via CLI for backward compatibility.
    parser.add_argument("--quant_func", type=str, default=None)
    parser.add_argument("--quant_mod", type=str, default=None)
    parser.add_argument("--num_expert", type=int, default=None)
    parser.add_argument("--is_po2", type=str2bool, default=None)
    parser.add_argument("--residual", type=str2bool, default=None)
    parser.add_argument("--split_dim", type=int, default=None)
    parser.add_argument("--eff_bit", type=float, default=None)
    parser.add_argument("--kv_factor", type=float, default=None)
    parser.add_argument("--use_itq", type=str2bool, default=None)
    parser.add_argument("--itq_n_iter", type=int, default=None)

    # Evaluation args
    parser.add_argument("--ppl_task", type=str, default="wikitext2")
    parser.add_argument("--zeroshot_task", type=str,
                        default="boolq,piqa,hellaswag,winogrande,arc_easy,arc_challenge,openbookqa")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_fewshot", type=int, default=0)
    parser.add_argument("--limit", type=int, default=-1)
    parser.add_argument("--seqlen", type=int, default=None,
                        help="Sequence length for PPL evaluation (default: auto-detect from config)")

    args = parser.parse_args()
    main(args)
