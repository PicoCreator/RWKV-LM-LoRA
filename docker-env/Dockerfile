# Built on existing nvidia cuda image
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04

# Disable interactive prompt (with tzdata)
ARG DEBIAN_FRONTEND=noninteractive

# Install python3.11 deps, project dependencies, and pip3
RUN DEBIAN_FRONTEND=noninteractive apt-get update && \
    apt-get install -y build-essential checkinstall \ 
    libreadline-gplv2-dev  libncursesw5-dev libssl-dev \
    libsqlite3-dev tk-dev libgdbm-dev libc6-dev libbz2-dev \
    ninja-build software-properties-common python-is-python3 \
    curl wget vim bash

# Install python3.11
RUN add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y python3.11 python3.11-distutils python3.11-dev

# Update alternatives to use python3.11
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1

# Install pip for 3.11
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11

# Install the various dependencies
# they are split apart intentionally into the listed stages, 
# to resolve common installation / dependency issues seperately
# torch installed is for cuda 11.8
RUN python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
RUN python -m pip install datasets transformers 
RUn python -m pip install lightning==2.0.2 deepspeed==0.9.3
RUN python -m pip install ninja numexpr jsonargparse 'jsonargparse[signatures]' \
                          lm-dataformat ftfy sentencepiece tokenizers wandb rwkv

# Install nvidia cublas, and etc, requried for multi-gpu training
RUN apt install -y nvidia-cuda-toolkit 