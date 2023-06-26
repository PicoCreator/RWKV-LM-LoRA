# RWKV Memory Experiment

The following repo is ongoing repo for RWKV memory size experiments. And is not meant to be used directly.

If you are looking for the RWKV infctx trainer, refer to the original repo infctx branch here : https://github.com/Blealtan/RWKV-LM-LoRA/tree/dev-infctx

## Environment setup

The following venv setup using conda, modify for your use case respectively
```bash
# ninja-build is required for the new trainer
sudo apt-get install ninja-build

# Ensure conda is up to date
conda update conda

# Virtual env, with python 3.11
conda create -n rwkv-exp python=3.11 pip
conda activate rwkv-exp

# Install pytorch
conda install -y pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia

# We use python -m pip, instead of pip directly, as it resolve issues with venv not loading the right pip
python -m pip install datasets transformers 
python -m pip install lightning==2.0.2 deepspeed==0.9.3 
python -m pip install ninja numexpr jsonargparse 'jsonargparse[signatures]'
python -m pip install lm-dataformat ftfy sentencepiece tokenizers wandb
```

Due to issues with [deepspeed on windows](https://github.com/microsoft/DeepSpeed/issues/2427). Only linux environments are supported. WSl2 with windows is not recommended, due to heavy performance penalities in the process (cannot use deepspeed offload, ~50% slower)

## How was the init model built

Because I do not have enough vram to run training on the [main RWKV-LM trainer](https://github.com/BlinkDL/RWKV-LM/tree/main/RWKV-v4neo), the approach was to simply run the train process, let it init a model, crash. And take the model from there.

For example for model-A i can use any pre-existing tokenized file (important as it detects the vocab size) and do the following

```bash
# Adjust the datafile, layers, and embeding accordingly
python train.py --load_model "" --proj_dir "out-A" \
--data_file "../data/enwik8.npy" --data_type "numpy" --vocab_size 50277 \
--ctx_len 2048 --epoch_steps 5000 --epoch_count 20 --epoch_begin 0 --epoch_save 1 \
--micro_bsz 64 --n_layer 24 --n_embd 2048 --pre_ffn 0 --head_qk 0 \
--lr_init 8e-4 --lr_final 2e-4 --warmup_steps 0 --beta1 0.9 --beta2 0.99 --adam_eps 1e-8 \
--accelerator gpu --devices 1 --precision bf16 --strategy deepspeed_stage_2_offload --grad_cp 0
```
