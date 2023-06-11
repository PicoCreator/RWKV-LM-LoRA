# Install setup

```bash
conda create -n rwkv-exp python=3.11 pip
conda activate rwkv-exp

pip install lightning==2.0.2 deepspeed==0.9.3 torch==2.0.1 datasets transformers 
pip install numexpr jsonargparse 'jsonargparse[signatures]'


# pip install numexpr
# pip install lightning['extra']
# pip install lm-dataformat ftfy sentencepiece tokenizers
# pip install pytorch-lightning
```