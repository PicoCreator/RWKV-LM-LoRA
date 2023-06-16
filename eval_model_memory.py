#!/usr/bin/env python3
import sys
import os
import difflib

#---
# Given the RWKV model path
# Evaluate token memorization capabilities of the model
#
# Runs on GPU
#---

# Check for argument, else throw error
if len(sys.argv) < 2:
    print("No arguments supplied")
    print("Usage: python3 eval_model_memory.py <rwkv_model_path>")
    sys.exit(1)

# set these before import RWKV
os.environ['RWKV_JIT_ON'] = '1'
os.environ["RWKV_CUDA_ON"] = '1' # '1' to compile CUDA kernel (10x faster), requires c++ compiler & cuda libraries

from rwkv.model import RWKV
from rwkv.utils import PIPELINE
from rwkv.utils import PIPELINE_ARGS

# Model strategy to use
# model_run_strat='cpu fp32' # CPU only, use if you dun have a GPU
model_run_strat='cuda fp16' # Entire model is in the GPU (use if you have enough vram)
# model_run_strat='cuda fp16 *20+' # GPU streaming, if you have vram issues for 14B model
# model_run_strat='cuda fp16 *0+' # GPU streaming, if you have really low vram

# download models: https://huggingface.co/BlinkDL
model_path = sys.argv[1]
model = RWKV(model=model_path, strategy=model_run_strat)
pipeline = PIPELINE(model, "./RWKV-v4neo/20B_tokenizer.json") # 20B_tokenizer.json is in https://github.com/BlinkDL/ChatRWKV

# Get the cursed " on" token
on_token = pipeline.encode(" on")[0]
markdwon_token = pipeline.encode("```")[0]

# Pipeline args to use
pipeline_args = PIPELINE_ARGS(
                     temperature = 0.2, top_p = 0.2, 
                     top_k = 1, # top_k = 0 then ignore
                     alpha_frequency = 0,
                     alpha_presence = 0,
                     token_ban = [on_token], # ban the generation of some tokens
                     token_stop = [0,markdwon_token], # stop generation whenever you see any token here
                     chunk_len = 256) # split input into chunks to save VRAM (shorter -> slower)

# Read the test word list, taken from ./eval_word_list.txt
with open('./eval_word_list.txt', 'r') as f:
    test_word_list = f.read()

# Convert it to tokens
test_word_tokens = pipeline.encode(test_word_list)

# Prompt template prefix to use
prompt_prefix = "Memorise and reply back with the following document:"
completion_prefix = "Reply:"

# Function use to get words with the following token count
def get_words_with_token_count(token_count):
    target_tokens = test_word_tokens[:token_count]
    target_words = pipeline.decode(target_tokens)
    
    # Normalize to lowercase
    target_words = target_words.lower()

    return target_words

# Function for validating once the model at a specific token count
def validate_model(token_count):
    target_words = get_words_with_token_count(token_count)
    prompt = prompt_prefix + "\n```\n" + target_words + "\n```\n\n" + completion_prefix + "\n```\n"

    # Generate the completion
    gen_tokens = round(token_count * 1.5)
    completion = pipeline.generate(prompt, token_count=gen_tokens, args=pipeline_args)

    # Trim the target words of starting and ending spaces
    target_words = target_words.strip()

    # Split using "```" as the delimiter the completion
    completion = completion.split("```")[0].strip()

    # Get the similarity between the target words and the completion
    sm = difflib.SequenceMatcher(None, target_words, completion)
    similarity = sm.ratio()
    char_diff_count = len(target_words) + len(completion) - 2 * int(similarity * (len(target_words) + len(completion)) / 2)

    # Print the results
    print("=============")
    print(f'Model validation at {token_count} tokens : {similarity * 100}% similarity, {char_diff_count} char diff')

    # Print more info if there are differences
    if(char_diff_count > 0):
        print("---   target   ---")
        print(target_words)
        print("--- completion ---")
        print(completion)
        print("------------------")

# Print the start of model validation
print("")
print("### Model validation start ###")

# Validate the model at different token counts
validate_model(5)
validate_model(10)
validate_model(15)
validate_model(20)
validate_model(25)
validate_model(30)
validate_model(35)
validate_model(40)
validate_model(45)
validate_model(50)
validate_model(55)
validate_model(60)
validate_model(65)
validate_model(70)
validate_model(100)
# validate_model(200)
# validate_model(300)
# validate_model(400)
# validate_model(600)
# validate_model(700)
# validate_model(800)
# validate_model(900)
# validate_model(1000)
