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

from quantization.utils import get_quant_func_and_mod, patch_inst
from utils.datautils import get_qat_dataset
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


def prepare_dataset(args, tokenizer):
    text = tokenizer.__repr__()
    hash_key = re.sub(r"name_or_path=[^,]+,?\s*", "", text)

    hash_value = hashlib.sha256(hash_key.encode()).hexdigest()[:7]
    dataset = os.path.join(args.data_root, args.dataset, hash_value)

    logger.info(f"Attempting to load dataset from disk at '{dataset}'")
    try:
        datasets = load_from_disk(dataset)
        logger.info(f"Successfully loaded dataset from disk at '{dataset}'")
    except (FileNotFoundError, OSError) as e:
        logger.warning(f"Failed to load dataset from disk at '{dataset}': {e}")
        logger.info("Generating new dataset using get_qat_dataset")
        datasets = get_qat_dataset(args.dataset, tokenizer)
        datasets.save_to_disk(dataset)
        logger.info(f"Dataset saved to disk at '{dataset}'")
        with open(os.path.join(dataset, "tokenizer_info"), "w") as f:
            f.write(hash_key)
    return datasets


def load_tokenizer(model_id):
    return AutoTokenizer.from_pretrained(model_id, use_fast=False, trust_remote_code=True)


def load_student_model(args, device_map, torch_dtype):
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch_dtype,
        device_map="cpu",
        low_cpu_mem_usage=True,
    )
    model.config.use_cache = False
    prepare_model_for_training(model)

    quant_func, quant_mod = get_quant_func_and_mod(args.quant_func, args.quant_mod)

    KV_PATTERN = [re.compile(r'\.k_proj$'), re.compile(r'\.v_proj$')]
    mapping = {nn.Linear: quant_mod}

    convert_kwargs = [
        ([nn.Linear], {
            "do_train": True,
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
        convert_kwargs.append(([Phi3Attention], {'config': model.config}))

    patch_inst(
        model,
        convert_kwargs=convert_kwargs,
        mapping=mapping,
        exclude_names=["lm_head"],
        device_map=device_map,
    )

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


if __name__ == "__main__":
    main()
