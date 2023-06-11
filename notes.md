# Install setup

```bash
conda create -n rwkv-exp python=3.11 pip
conda activate rwkv-exp
conda install pip

# We use python -m pip, instead of pip directly, as it resolve issues with venv not loading the right pip
python -m pip install lightning==2.0.2 deepspeed==0.9.3 torch==2.0.1 datasets transformers 
python -m pip install numexpr jsonargparse 'jsonargparse[signatures]'
python -m pip install lm-dataformat ftfy sentencepiece tokenizers
python -m pip install wandb

# pip install numexpr
# pip install lightning['extra']
# pip install lm-dataformat ftfy sentencepiece tokenizers
# pip install pytorch-lightning
# pip install 'jsonargparse[signatures]==4.17.0'
```

Download the 1B5 model

```
mkdir -p model
cd model
wget https://huggingface.co/BlinkDL/rwkv-4-pile-1b5/resolve/main/RWKV-4b-Pile-1B5-20230217-7954.pth
wget https://huggingface.co/BlinkDL/rwkv-4-pile-169m/resolve/main/RWKV-4-Pile-169M-20220807-8023.pth
```

Run the model

```
python new_train.py fit -c ./config-169M.yaml 
```