<div align="center">

<h1>LittleBit: Ultra Low-Bit Quantization<br>via Latent Factorization</h1>

<h3>Banseok Lee<sup>*</sup>, Dongkyu Kim<sup>*</sup>, Youngcheon You, Youngmin Kim<sup>&dagger;</sup></h3>
<p><sup>*</sup>Equal Contribution, <sup>&dagger;</sup>Corresponding Author</p>

[![arXiv](https://img.shields.io/badge/arXiv-2506.13771-b31b1b.svg)](https://arxiv.org/abs/2506.13771)
[![NeurIPS](https://img.shields.io/badge/NeurIPS-2025-blue.svg)](https://neurips.cc/)

</div>

---

## 📢 Abstract

> **LittleBit** is a novel method for extreme LLM compression, targeting levels like **0.1 bits per weight (BPW)**. By representing weights in a low-rank form using latent matrix factorization and subsequently binarizing these factors, it achieves nearly **31× memory reduction** (e.g., Llama2-13B to under 0.9 GB). To counteract information loss, it integrates a multi-scale compensation mechanism including row, column, and an additional latent dimension learning per-rank importance.

---

## ✨ Key Features

### 🧠 Model Architecture & Support
* **Extreme Compression:** Targets **0.1 BPW** regime.
* **High Efficiency:** **31× memory reduction** compared to FP16.
* **Novel Method:** Latent Matrix Factorization with Binarization & Multi-scale Compensation.

### 🏗️ Supported Models
The codebase currently supports the following architectures:
* ✅ **OPT**
* ✅ **Llama** (Llama-2, Llama-3)
* ✅ **Phi-4**
* ✅ **Qwen2.5 (QwQ)**
* ✅ **Gemma 2** & **Gemma 3**

---

## 💿 Installation

Set up the environment using Conda and Pip. We recommend using Python 3.12.

```bash
conda create -n littlebit python=3.12
conda activate littlebit

# Install CUDA toolkit (adjust version as necessary)
conda install nvidia/label/cuda-12.4.1::cuda-toolkit -c nvidia/label/cuda-12.4.1

# Install PyTorch
pip install torch==2.8.0+cu124 torchvision==0.23.0+cu124 torchaudio==2.8.0+cu124 --index-url https://download.pytorch.org/whl/cu124

# Install dependencies
pip install -r requirements.txt
````

-----

## 🚀 Usage

### 1\. Training (QAT)

Train the model using Quantization-Aware Training (QAT) with the LittleBit approach.

**Single GPU Example:**

```bash
CUDA_VISIBLE_DEVICES=0 python -m main \
    --model_id meta-llama/Llama-2-7b-hf \
    --dataset c4_wiki \
    --save_dir ./outputs/Llama-2-7b-LittleBit \
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
    --l2l_loss_scale 10.0
```

**Multi-GPU (DeepSpeed) Example:**

```bash
deepspeed --num_gpus=4 main.py \
    --model_id meta-llama/Llama-2-7b-hf \
    --dataset c4_wiki \
    --save_dir ./outputs/Llama-2-7b-LittleBit \
    --ds_config_path configs/ds_config.json \
    --num_train_epochs 5.0 \
    --per_device_train_batch_size 4 \
    --lr 4e-05 \
    --report wandb \
    --quant_func SmoothSign \
    --quant_mod LittleBitLinear \
    --residual True \
    --eff_bit 1.0
```

### 2\. Evaluation

Evaluate the trained LittleBit model on Perplexity (PPL) tasks and Zero-shot benchmarks.

```bash
CUDA_VISIBLE_DEVICES=0 python -m eval \
    --model_type llama \
    --model_id ./outputs/Llama-2-7b-LittleBit \
    --quant_func SmoothSign \
    --quant_mod LittleBitLinear \
    --residual True \
    --eff_bit 1.0 \
    --kv_factor 1.0 \
    --ppl_task wikitext2,c4 \
    --zeroshot_task boolq,piqa,hellaswag,winogrande,arc_easy,arc_challenge,openbookqa
```

-----

## 📝 Citation

If you find this work useful, please cite our paper:

```bibtex
@inproceedings{littlebit,
  title={LittleBit: Ultra Low-Bit Quantization via Latent Factorization},
  author={Lee, Banseok and Kim, Dongkyu and You, Youngcheon and Kim, Youngmin},
  booktitle={Advances in Neural Information Processing Systems},
  year={2025},
}
```

## ⚖️ License

This project is licensed under the [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) license.
