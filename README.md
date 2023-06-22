# RWKV Memory Experiment

The following repo is ongoing repo for RWKV memory size experiments. And is not meant to be used directly.

If you are looking for the RWKV infctx trainer, refer to the original repo infctx branch here : https://github.com/Blealtan/RWKV-LM-LoRA/tree/dev-infctx

## Environment setup

The following venv setup using conda, modify for your use case respectively
```
# ninja-build is required for the new trainer
sudo apt-get install ninja-build

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