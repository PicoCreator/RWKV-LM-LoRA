# Install setup

conda create -n rwkv-exp python=3.11 pip
conda activate rwkv-exp

pip install lightning==2.0.2 deepspeed==0.9.3 torch==2.0.1 cudatoolkit=11.7 cudatoolkit-dev=11.7 datasets transformers 
pip install 'jsonargparse[signatures]>=4.17.0'
pip install numexpr

pip install lightning['extra']
pip install lm-dataformat ftfy sentencepiece tokenizers
pip install pytorch-lightning