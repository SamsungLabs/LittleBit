<div align="center">

# The LittleBit Project

### Sub-1-Bit LLM Compression via Latent Factorization

Official implementation of **LittleBit** (NeurIPS 2025) and **LittleBit-2** (ICML 2026).

</div>

---

## Papers

**LittleBit-2: Maximizing the Spectral Energy Gain in Sub-1-Bit LLMs via Latent Geometry Alignment** *(ICML 2026)*<br>
Banseok Lee, Youngmin Kim<br>
[![arXiv](https://img.shields.io/badge/arXiv-2603.00042-b31b1b.svg)](https://arxiv.org/abs/2603.00042)
[![ICML](https://img.shields.io/badge/ICML-2026-blue.svg)](https://icml.cc/)

**LittleBit: Ultra Low-Bit Quantization via Latent Factorization** *(NeurIPS 2025)*<br>
Banseok Lee*, Dongkyu Kim*, Youngcheon You, Youngmin Kim<br>
[![arXiv](https://img.shields.io/badge/arXiv-2506.13771-b31b1b.svg)](https://arxiv.org/abs/2506.13771)
[![NeurIPS](https://img.shields.io/badge/NeurIPS-2025-blue.svg)](https://neurips.cc/)

---

## Abstract

**LittleBit** compresses large language models into the sub-1-bit regime by factorizing each dense weight matrix into low-rank latent factors, binarizing those factors, and restoring magnitude information through lightweight learned scales. This enables extreme compression, including the 0.1 bits-per-weight setting, while preserving the original model architecture at inference time.

**LittleBit-2** improves this recipe by addressing latent geometry misalignment in the initialization stage. It applies Internal Latent Rotation with Joint Iterative Quantization (Joint-ITQ), aligning the SVD-derived latent factors with the binary hypercube before QAT. The implementation in this repository now applies LittleBit-2 initialization by default for `LittleBitLinear`, with no additional inference overhead.

---

## Highlights

- **Sub-1-bit compression:** Designed for 1.0 to 0.1 bits per weight.
- **LittleBit-2 by default:** `LittleBitLinear` uses Joint-ITQ initialization automatically.
- **No inference-time change:** LittleBit-2 modifies initialization only; the deployed factorized layer remains the same.
- **QAT-friendly:** Supports Quantization-Aware Training with SmoothSign and optional residual factorization.

## Supported Models

The codebase currently supports:

- OPT
- Llama and Llama 2/3
- Phi-4
- Qwen2.5 and QwQ
- Gemma 2 and Gemma 3
- Qwen3

---

## Installation

We recommend Python 3.12.

```bash
conda create -n littlebit python=3.12
conda activate littlebit

# Install CUDA toolkit. Adjust the CUDA version if needed.
conda install nvidia/label/cuda-12.4.1::cuda-toolkit -c nvidia/label/cuda-12.4.1

# Install PyTorch.
pip install torch==2.8.0+cu124 torchvision==0.23.0+cu124 torchaudio==2.8.0+cu124 --index-url https://download.pytorch.org/whl/cu124

# Install dependencies.
pip install -r requirements.txt
```

> [!IMPORTANT]
> For reproducing the paper results, use `transformers` 4.51.x. Newer `transformers` releases may change model internals or evaluation behavior.

```bash
pip install "transformers==4.51.*"
```

---

## Usage

### Training

Train a model with Quantization-Aware Training. LittleBit-2 is the default initialization path when using `--quant_mod LittleBitLinear`.

**Single GPU**

```bash
CUDA_VISIBLE_DEVICES=0 python -m main \
    --model_id meta-llama/Llama-2-7b-hf \
    --dataset c4_wiki \
    --save_dir ./outputs/Llama-2-7b-LittleBit-2 \
    --num_train_epochs 5.0 \
    --per_device_train_batch_size 4 \
    --lr 4e-05 \
    --warmup_ratio 0.02 \
    --report wandb \
    --quant_func SmoothSign \
    --quant_mod LittleBitLinear \
    --residual True \
    --eff_bit 1.0 \
    --kv_factor 1.0 \
    --min_split_dim 8 \
    --l2l_loss_scale 10.0
```

**Multi-GPU with DeepSpeed**

```bash
deepspeed --num_gpus=4 main.py \
    --model_id meta-llama/Llama-2-7b-hf \
    --dataset c4_wiki \
    --save_dir ./outputs/Llama-2-7b-LittleBit-2 \
    --ds_config_path configs/zero3.json \
    --num_train_epochs 5.0 \
    --per_device_train_batch_size 4 \
    --lr 4e-05 \
    --report wandb \
    --quant_func SmoothSign \
    --quant_mod LittleBitLinear \
    --residual True \
    --eff_bit 1.0 \
    --kv_factor 1.0 \
    --min_split_dim 8
```

### Evaluation

Evaluate a local checkpoint or a model hosted on the Hugging Face Hub.

```bash
# From a local directory
CUDA_VISIBLE_DEVICES=0 python eval.py \
    --model_id ./outputs/Llama-2-7b-LittleBit-2 \
    --seqlen 2048 \
    --ppl_task wikitext2,c4 \
    --zeroshot_task boolq,piqa,hellaswag,winogrande,arc_easy,arc_challenge,openbookqa

# From the Hugging Face Hub
CUDA_VISIBLE_DEVICES=0 python eval.py \
    --model_id username/littlebit-llama-7b-0.1bpw \
    --seqlen 2048 \
    --ppl_task wikitext2
```

### Legacy Checkpoints

Older checkpoints may not include `littlebit_config.json`. In that case, pass the quantization arguments explicitly:

```bash
CUDA_VISIBLE_DEVICES=0 python eval.py \
    --model_id ./outputs/Legacy-Llama-2-7b \
    --quant_func SmoothSign \
    --quant_mod LittleBitLinear \
    --split_dim 1024
```

Parameter loading priority:

1. Explicit CLI arguments
2. `littlebit_config.json` in the model directory
3. `config.json` fallback for older checkpoints

---

## Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{lee2026littlebit2,
  title={LittleBit-2: Maximizing the Spectral Energy Gain in Sub-1-Bit LLMs via Latent Geometry Alignment},
  author={Lee, Banseok and Kim, Youngmin},
  booktitle={Proceedings of the 43rd International Conference on Machine Learning},
  year={2026}
}
```

```bibtex
@inproceedings{lee2025littlebit,
  title={LittleBit: Ultra Low-Bit Quantization via Latent Factorization},
  author={Lee, Banseok and Kim, Dongkyu and You, Youngcheon and Kim, Youngmin},
  booktitle={Advances in Neural Information Processing Systems},
  year={2025}
}
```

## License

This project is licensed under the [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) license.
