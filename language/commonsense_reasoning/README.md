# MambaPEFT for Language Tasks

This directory includes the MambaPEFT for reproducing the results in our paper.  

## Setup

Install dependencies. Our code is tested on CUDA 12.1 and Python 3.12.7.

```sh
# install packages
pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

## Datasets

Download the commonsense 170k fine-tuning dataset from [here](https://github.com/AGI-Edgerunners/LLM-Adapters/blob/main/ft-training_set/commonsense_170k.json), then place the data as follows
```bash
# Dataset for fine-tuning
launguage/commonsense_reasoning/commonsense_170k.json
```

## Fine-tuning

### Full fine-tuning

```sh
# Full fine-tuning
python finetune.py --base_model state-spaces/mamba-130m-hf --data_path commonsense_170k.json --adapter_name full --output_dir ./results/full --learning_rate 5e-5
```

### Parameter-efficient fine-tuning(PEFT)

Currently supports fine-tuning on single GPU.

```sh
# Additional-scan
CUDA_VISIBLE_DEVICES=0 python finetune.py --base_model state-spaces/mamba-130m-hf --data_path commonsense_170k.json --adapter_name AddiScan --output_dir ./results/addiscan --learning_rate 5e-3

# Affix-tuning
CUDA_VISIBLE_DEVICES=0 python finetune.py --base_model state-spaces/mamba-130m-hf --data_path commonsense_170k.json --adapter_name AffixTuning --output_dir ./results/affix --learning_rate 1e-4

# Apply LoRA on 'X' in Mamba
CUDA_VISIBLE_DEVICES=0 python finetune.py --base_model state-spaces/mamba-130m-hf --data_path commonsense_170k.json --adapter_name lora_X --output_dir ./results/lora_X --learning_rate 1e-3
```

## Evaluation

We use [lm_eval_harness](https://github.com/EleutherAI/lm-evaluation-harness) for evaluation.


```sh
TASKS='boolq,social_iqa,hellaswag,piqa,arc_easy,arc_challenge,winogrande,openbookqa'

# Zero-shot evaluation (Without fine-tuning)
lm_eval --model hf --model_args pretrained="state-spaces/mamba-130m-hf,trust_remote_code=True" --tasks $TASKS trust_remote_code=True

# PEFT evaluation
python lm_harness_eval.py --model MambaPEFT --model_args pretrained="state-spaces/mamba-130m-hf,peft_weights=./results/lora_X,trust_remote_code=True" --output_path results/lora_X --tasks $TASKS
```

## Acknowledgements

Our code is based on [Mamba](https://github.com/state-spaces/mamba), [LLM-Adapters](https://github.com/AGI-Edgerunners/LLM-Adapters), [PEFT](https://github.com/huggingface/peft), and [Transformers](https://github.com/huggingface/transformers).
