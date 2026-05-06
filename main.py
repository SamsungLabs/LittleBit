import re
import hashlib
import argparse
import datetime
import json
import os
from pathlib import Path

import deepspeed
import GPUtil
import torch
import torch.nn as nn
from datasets import load_from_disk
from transformers import default_data_collator
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, set_seed
from transformers.integrations.deepspeed import HfDeepSpeedConfig

from quantization.utils import apply_littlebit_patch
from utils.datautils import prepare_dataset, load_tokenizer
from utils.kd_utils import KDTrainer
from utils.misc import setup_logger
from utils.utils import prepare_model_for_training, print_trainable_parameters

logger = setup_logger(__name__)


def get_device_config():
    gpus = GPUtil.getGPUs()
    if not gpus:
        return None, None

    device_map = "auto"
    local_rank_str = os.environ.get('LOCAL_RANK')
    if local_rank_str is not None:
        try:
            local_rank = int(local_rank_str)
            device_map = {'': local_rank}
        except ValueError:
            pass

    return len(gpus), device_map


def str2bool(value):
    if value.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif value.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise Exception(f'Boolean value expected: {value}')


def get_args():
    parser = argparse.ArgumentParser(description="Model Training Script")
    parser.add_argument("--model_id", type=str, default="meta-llama/Llama-2-7b-hf")
    parser.add_argument("--data_root", type=str, default="./")
    parser.add_argument("--dataset", type=str, default="c4_wiki", choices=['c4', 'wikitext2', 'c4_wiki'])
    parser.add_argument("--save_dir", type=str, default='outputs')
    parser.add_argument("--f_name", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42, help="Seed")
    parser.add_argument("--num_train_epochs", type=float, default=3)
    parser.add_argument("--per_device_train_batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--l2l_loss_scale", type=float, default=10.0)
    parser.add_argument("--dataset_prepared", type=str2bool, default=True)
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--ds_config_path", type=str, default=None)
    parser.add_argument("--exp_name", type=str, default="LittleBit")
    parser.add_argument("--run_name", type=str, default="my_run")
    parser.add_argument("--report", nargs="+", default=["wandb"], choices=["wandb", "tensorboard"])
    parser.add_argument("--quant_func", type=str, default="STEBinary")
    parser.add_argument("--quant_mod", type=str, default="LittleBitLinear")

    parser.add_argument("--residual", type=str2bool, default=False)
    parser.add_argument("--split_dim", type=int, default=1024)
    parser.add_argument("--eff_bit", type=float, default=1.0)
    parser.add_argument("--kv_factor", type=float, default=1.0)
    parser.add_argument("--min_split_dim", type=int, default=8)

    parser.add_argument("--use_itq", type=str2bool, default=False)
    parser.add_argument("--itq_n_iter", type=int, default=50)

    args = parser.parse_args()

    return args


def get_save_dir(args):
    if args.save_dir is None:
        raise ValueError("save_dir cannot be None")

    f_name = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M') if args.f_name is None else args.f_name
    save_dir = os.path.join(args.save_dir, f_name)
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    return save_dir


def get_training_arguments(args, save_dir):
    return TrainingArguments(
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_ratio=args.warmup_ratio,
        num_train_epochs=args.num_train_epochs,
        bf16=True,
        logging_steps=1,
        save_strategy="no",
        save_steps=10000,
        output_dir=save_dir,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        optim="adamw_torch",
        adam_beta1=args.adam_beta1,
        adam_beta2=args.adam_beta2,
        deepspeed=args.ds_config_path,
        report_to=args.report,
        run_name=args.run_name,
    )


def load_student_model(args, device_map, torch_dtype):
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch_dtype,
        device_map="cpu",
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    model.config.use_cache = False
    prepare_model_for_training(model)

    print("INFO: Applying quantization patch...")
    model = apply_littlebit_patch(model, args, do_train=True)

    if device_map:
        model.to(device_map if isinstance(device_map, (str, torch.device)) else list(device_map.values())[0])

    print_trainable_parameters(model)
    return model


def load_teacher_model(args, num_gpus, torch_dtype, config_path="configs/zero3_inference.json"):
    with open(config_path, 'r') as f:
        config = json.load(f)

    _ = HfDeepSpeedConfig(config)

    teacher_model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch_dtype,
    )
    teacher_model.eval()
    for param in teacher_model.parameters():
        param.requires_grad = False
    teacher_model.config.use_cache = False

    teacher_model, _, _, _ = deepspeed.initialize(
        model=teacher_model,
        model_parameters=teacher_model.parameters(),
        config=config,
    )

    return teacher_model


def setup_trainer(model, teacher_model, tokenizer, datasets, training_args, args):
    trainer = KDTrainer(
        model=model,
        teacher_model=teacher_model,
        l2l_loss_scale=args.l2l_loss_scale,
        processing_class=tokenizer,
        train_dataset=datasets,
        args=training_args,
        data_collator=default_data_collator,
    )
    return trainer


def save_artifacts(trainer, model, tokenizer, save_dir, args):
    try:
        model.eval()
        model.config.use_cache = True
        for param in model.parameters():
            param.data = param.data.to(torch.bfloat16)
        trainer.save_model(output_dir=save_dir)
        logger.info(f"Model saved to {save_dir}")

        tokenizer.save_pretrained(save_dir)
        logger.info(f"Model and tokenizer saved to {save_dir}")

    except Exception as save_err:
        logger.error(f"Failed during final save/log: {save_err}", exc_info=True)


def main():
    args = get_args()
    set_seed(args.seed)

    save_dir = get_save_dir(args)

    num_gpus, device_map = get_device_config()

    logger.info("Loading tokenizer...")
    tokenizer = load_tokenizer(args.model_id)

    logger.info(f"Preparing training data ({args.dataset})...")
    datasets = prepare_dataset(args, tokenizer)

    logger.info("Loading student model...")
    model = load_student_model(args, device_map, torch.bfloat16)

    logger.info(f"Loading teacher model...")
    teacher_model = load_teacher_model(args, num_gpus, torch.bfloat16)

    training_args = get_training_arguments(args, save_dir)

    logger.info(f"Setting trainer...")
    trainer = setup_trainer(model, teacher_model, tokenizer, datasets, training_args, args)

    trainer.train()

    save_artifacts(trainer, model, tokenizer, save_dir, args)


def save_artifacts(trainer, model, tokenizer, save_dir, args):
    try:
        logger.info("Starting artifact saving process (Grouped Chunk Strategy)...")

        if hasattr(trainer, 'accelerator'):
            unwrapped_model = trainer.accelerator.unwrap_model(model)
        else:
            unwrapped_model = model
            while hasattr(unwrapped_model, 'module'):
                unwrapped_model = unwrapped_model.module

        use_ds = (args.ds_config_path is not None)
        final_cpu_state_dict = {}

        if use_ds:
            logger.info("DeepSpeed ZeRO-3 enabled. Gathering parameters in groups...")

            LAYER_CHUNK_SIZE = 4
            for name, module in unwrapped_model.named_children():
                if isinstance(module, torch.nn.ModuleList):
                    num_layers = len(module)
                    for i in range(0, num_layers, LAYER_CHUNK_SIZE):
                        end_idx = min(i + LAYER_CHUNK_SIZE, num_layers)
                        layer_group = module[i:end_idx]

                        logger.info(f"Gathering layers {i} to {end_idx-1}...")

                        with deepspeed.zero.GatheredParameters(layer_group.parameters(), modifier_rank=0):
                            if args.local_rank == 0 or args.local_rank == -1:
                                for idx, layer in enumerate(layer_group):
                                    layer_global_idx = i + idx
                                    layer_state_dict = layer.state_dict()
                                    for k, v in layer_state_dict.items():
                                        final_cpu_state_dict[f"{name}.{layer_global_idx}.{k}"] = v.cpu()

                else:
                    logger.info(f"Processing module: {name}")
                    with deepspeed.zero.GatheredParameters(module.parameters(), modifier_rank=0):
                        if args.local_rank == 0 or args.local_rank == -1:
                            module_state_dict = module.state_dict()
                            for k, v in module_state_dict.items():
                                final_cpu_state_dict[f"{name}.{k}"] = v.cpu()

            remaining_params = [p for n, p in unwrapped_model.named_parameters() if '.' not in n]
            if remaining_params:
                with deepspeed.zero.GatheredParameters(remaining_params, modifier_rank=0):
                    if args.local_rank == 0 or args.local_rank == -1:
                        for n, p in unwrapped_model.named_parameters():
                            if '.' not in n:
                                final_cpu_state_dict[n] = p.cpu()
        else:
            final_cpu_state_dict = {k: v.cpu() for k, v in unwrapped_model.state_dict().items()}

        if args.local_rank == 0 or args.local_rank == -1:
            logger.info("Saving to disk...")

            # Automatically save quantization parameters in config
            quant_params = {
                "quant_func": getattr(args, "quant_func", "STEBinary"),
                "eff_bit": getattr(args, "eff_bit", 1.0),
                "split_dim": getattr(args, "split_dim", 1024),
                "residual": getattr(args, "residual", False),
                "kv_factor": getattr(args, "kv_factor", 1.0),
                "min_split_dim": getattr(args, "min_split_dim", 8),
                "quant_mod": getattr(args, "quant_mod", "LittleBitLinear"),
                "use_itq": getattr(args, "use_itq", False),
                "itq_n_iter": getattr(args, "itq_n_iter", 50),
            }

            littlebit_config_path = os.path.join(save_dir, "littlebit_config.json")
            with open(littlebit_config_path, "w", encoding="utf-8") as f:
                json.dump(quant_params, f, indent=2)
            logger.info(f"Saved LittleBit config to {littlebit_config_path}")

            for key, value in quant_params.items():
                setattr(unwrapped_model.config, key, value)

            unwrapped_model.config.use_cache = True

            for k, v in final_cpu_state_dict.items():
                if "packed" not in k and "shape" not in k and v.dtype == torch.float32:
                    final_cpu_state_dict[k] = v.to(torch.bfloat16)

            unwrapped_model.save_pretrained(save_dir, state_dict=final_cpu_state_dict, safe_serialization=True)
            tokenizer.save_pretrained(save_dir)

            logger.info("Artifacts saved successfully.")
            del final_cpu_state_dict
            import gc
            gc.collect()

    except Exception as save_err:
        logger.error(f"Failed during final save/log: {save_err}", exc_info=True)


if __name__ == "__main__":
    main()
