# How to replicate the memory experiment

Clone the project into `/home/ubuntu/RWKV-LM-LoRA`
If the dir differ, you will need to change the settings in `RWKV-v4neo/config-3B.yaml`

Also if your GPU and system specs differ from mine (it probably does), MODIFY the config-3B file accordingly

## Setup the conda env

```bash
conda create -n rwkv-exp python=3.11 pip
conda activate rwkv-exp

# We use python -m pip, instead of pip directly, as it resolve issues with venv not loading the right pip
python -m pip install lightning==2.0.2 deepspeed==0.9.3 torch==2.0.1 datasets transformers 
python -m pip install ninja numexpr jsonargparse 'jsonargparse[signatures]'
python -m pip install lm-dataformat ftfy sentencepiece tokenizers
python -m pip install wandb

# For running the eval test
python -m pip install rwkv
```

## Setup the required directories & download the required model

```bash
# All the key dir setup
./setup-dir.sh

# Download the 3B model
cd model
wget https://huggingface.co/BlinkDL/rwkv-4-raven/resolve/main/RWKV-4-Raven-3B-v12-Eng98%25-Other2%25-20230520-ctx4096.pth
# wget https://huggingface.co/BlinkDL/rwkv-4-raven/resolve/main/RWKV-4-Raven-7B-v12-Eng98%25-Other2%25-20230521-ctx8192.pth

cd ..
```

## Generate the dataset

```bash
# Install NPM dependencies
npm install

# Generate the dataset
./gen_dataset.sh
```

## Perform the finetune

```bash
cd RWKV-v4neo

# Note you may want to comment out the logging in config-3B if you do not have weights & bias setup
python3 new_train.py fit -c ./config-3B.yaml 

# After the finetune export the checkpoint
# NOTE: you will need to modify the file path to your actual checkpoint
python3 export_checkpoint.py ../checkpoint/epoch=0-step=180.ckpt/

# Move the built model out
mv ../checkpoint/epoch=0-step=180.ckpt/rwkv_model.pth ../build/telephone-3B.pth
```

## Perform the guided test

```bash
# Make sure you are in the proj folder
cd ..

# Run the guided eval for the built model
python3 eval_model_memory_guided.py ./build/telephone-3B.pth

# Or run it for the base model
python3 eval_model_memory_guided.py ./model/RWKV-4-Raven-3B-v12-Eng98%25-Other2%25-20230520-ctx4096.pth
```