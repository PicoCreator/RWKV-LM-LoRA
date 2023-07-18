#!/usr/bin/env python3
import sys, os, yaml
import torch

# ----
# This script is used to do an inference test through lightning api directly
# allowing us to perform inference, without rewriting a new "inference" library
#
# Mostly useful for performing experiments on the model
# ----

# Check for argument, else throw error
if len(sys.argv) < 2:
    print("No arguments supplied")
    print("Usage: python3 lightning_dragon_test.py <model-path>")
    sys.exit(1)
MODEL_PATH=sys.argv[1]

# ----
# Lets load the model
# ----

# Lets disable LightningModule
# os.environ["RWKV_USE_NN_MODULE"] = "1"
torch.set_default_dtype(torch.float32)

from src.model import RWKV
from src.trainer import RWKVLightningTrainer
from transformers import PreTrainedTokenizerFast

# Lets load the model directly, it has to be loaded to CPU first
# otherwise fp / bf related issues occurs
loaded_state = torch.load(MODEL_PATH, map_location='cpu')

# Get the model params
model_keys = list(loaded_state.keys())

# Get the maximum block id
max_block_id = 0
for x in model_keys:
    if 'blocks.' in x:
        block_id = int(x.split('.')[1])
        max_block_id = max(max_block_id, block_id)

# Compute the layer count & embed sizes
n_layer = max_block_id + 1
n_embd = loaded_state['head.weight'].shape[1]
vocab_size = loaded_state['head.weight'].shape[0]

# Context length used for block inference
# has a direct impact onto vram usage
CTX_LEN = 1024

## ---

# Prepare the model config with the model path, and custom torch load
model_config = {}
model_config["load_model"] = MODEL_PATH
model_config["n_embd"] = n_embd 
model_config["n_layer"] = n_layer 
model_config["vocab_size"] = vocab_size 
model_config["_torch_load_state"] = loaded_state
model_config["ctx_len"] = CTX_LEN

# Disable grad_cp, as it uses deepspeed
model_config["grad_cp"] = False

## ---

# Lets load the model, and set it to eval mode
model = RWKV(**model_config)
model.eval()

# Tokenizer (only support 20B for now), we also disable parallelism,
# as its not really needed, and induces lots of warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"
tokenizer_file = "./20B_tokenizer.json"
tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_file)

# Lets eval the model?
def _completion(
        prompt,
        token_count: int = 200,
        ):
    # Tokenize the prompt
    prompt_tokens_input = tokenizer(prompt, return_tensors="pt")["input_ids"].type(torch.long)
    prompt_tokens = prompt_tokens_input[0]
    prompt_tokens_len = len(prompt_tokens)

    # Get the model context len cutoff
    ctx_len_limit = int(model_config["ctx_len"])

    # Lets generate the initial logits and hidden states
    logits_arr = None
    last_shift_states = None
    last_wkv_states = None

    print("")

    # Lets loop through the prompt tokens, in chunks of ctx_len_limit
    for i in range(0, prompt_tokens_len, ctx_len_limit):
        # print("attempting inference pass")

        tokens = prompt_tokens_input[:, i:i+ctx_len_limit]
        print("tokens", tokens)

        logits_arr, last_shift_states, last_wkv_states = model(
            tokens, last_shift_states=last_shift_states, last_wkv_states=last_wkv_states
        )
        
        # # Log the logits?
        print( "logits.shape", logits_arr.shape )
        print( "logits dtype", logits_arr.dtype )
        print( "logits.??", logits_arr[0][0] )
        # print( "logits", logits )

    print( "prompt_tokens.shape", prompt_tokens.shape )

# Lets eval the model?
def completion(
        prompt,
        token_count: int = 200,
        ):
    with torch.no_grad():
        _completion(prompt, token_count=token_count)
    
# Perform the dragon test
prompt = "\nIn a shocking finding, scientist discovered a herd of dragons living in a remote, previously unexplored valley, in Tibet. Even more surprising to the researchers was the fact that the dragons spoke perfect Chinese."
# print(prompt, end='')
completion(prompt)

# # If model strategy is not specified, use 'cpu fp32' as default
# MODEL_STRATEGY=None
# if len(sys.argv) >= 3:
#     MODEL_STRATEGY=sys.argv[2]
# if MODEL_STRATEGY is None:
#     MODEL_STRATEGY = 'cpu fp32'

# # # set these before import RWKV
# # os.environ['RWKV_JIT_ON'] = '1'
# # os.environ["RWKV_CUDA_ON"] = '0' # '1' to compile CUDA kernel (10x faster), requires c++ compiler & cuda libraries

# ########################################################################################################
# #
# # Use '/' in model path, instead of '\'. Use ctx4096 models if you need long ctx.
# #
# # fp16 = good for GPU (!!! DOES NOT support CPU !!!)
# # fp32 = good for CPU
# # bf16 = worse accuracy, supports CPU
# # xxxi8 (example: fp16i8, fp32i8) = xxx with int8 quantization to save 50% VRAM/RAM, slower, slightly less accuracy
# #
# # We consider [ln_out+head] to be an extra layer, so L12-D768 (169M) has "13" layers, L24-D2048 (1.5B) has "25" layers, etc.
# # Strategy Examples: (device = cpu/cuda/cuda:0/cuda:1/...)
# # 'cpu fp32' = all layers cpu fp32
# # 'cuda fp16' = all layers cuda fp16
# # 'cuda fp16i8' = all layers cuda fp16 with int8 quantization
# # 'cuda fp16i8 *10 -> cpu fp32' = first 10 layers cuda fp16i8, then cpu fp32 (increase 10 for better speed)
# # 'cuda:0 fp16 *10 -> cuda:1 fp16 *8 -> cpu fp32' = first 10 layers cuda:0 fp16, then 8 layers cuda:1 fp16, then cpu fp32
# #
# # Basic Strategy Guide: (fp16i8 works for any GPU)
# # 100% VRAM = 'cuda fp16'                   # all layers cuda fp16
# #  98% VRAM = 'cuda fp16i8 *1 -> cuda fp16' # first 1 layer  cuda fp16i8, then cuda fp16
# #  96% VRAM = 'cuda fp16i8 *2 -> cuda fp16' # first 2 layers cuda fp16i8, then cuda fp16
# #  94% VRAM = 'cuda fp16i8 *3 -> cuda fp16' # first 3 layers cuda fp16i8, then cuda fp16
# #  ...
# #  50% VRAM = 'cuda fp16i8'                 # all layers cuda fp16i8
# #  48% VRAM = 'cuda fp16i8 -> cpu fp32 *1'  # most layers cuda fp16i8, last 1 layer  cpu fp32
# #  46% VRAM = 'cuda fp16i8 -> cpu fp32 *2'  # most layers cuda fp16i8, last 2 layers cpu fp32
# #  44% VRAM = 'cuda fp16i8 -> cpu fp32 *3'  # most layers cuda fp16i8, last 3 layers cpu fp32
# #  ...
# #   0% VRAM = 'cpu fp32'                    # all layers cpu fp32
# #
# # Use '+' for STREAM mode, which can save VRAM too, and it is sometimes faster
# # 'cuda fp16i8 *10+' = first 10 layers cuda fp16i8, then fp16i8 stream the rest to it (increase 10 for better speed)
# #
# # Extreme STREAM: 3G VRAM is enough to run RWKV 14B (slow. will be faster in future)
# # 'cuda fp16i8 *0+ -> cpu fp32 *1' = stream all layers cuda fp16i8, last 1 layer [ln_out+head] cpu fp32
# #
# # ########################################################################################################

# from rwkv.model import RWKV
# from rwkv.utils import PIPELINE, PIPELINE_ARGS

# # download models: https://huggingface.co/BlinkDL
# model = RWKV(model=MODEL_PATH, strategy=MODEL_STRATEGY)
# pipeline = PIPELINE(model, "./20B_tokenizer.json") # 20B_tokenizer.json is in https://github.com/BlinkDL/ChatRWKV

# prompt = "\nIn a shocking finding, scientist discovered a herd of dragons living in a remote, previously unexplored valley, in Tibet. Even more surprising to the researchers was the fact that the dragons spoke perfect Chinese."
# print(prompt, end='')

# def my_print(s):
#     print(s, end='', flush=True)

# # For alpha_frequency and alpha_presence, see "Frequency and presence penalties":
# # https://platform.openai.com/docs/api-reference/parameter-details

# args = PIPELINE_ARGS(temperature = 1.0, top_p = 0.7, top_k = 100, # top_k = 0 then ignore
#                      alpha_frequency = 0.25,
#                      alpha_presence = 0.25,
#                      #alpha_decay=0.996, # gradually decay the penalty
#                      token_ban = [0], # ban the generation of some tokens
#                      token_stop = [], # stop generation whenever you see any token here
#                      chunk_len = 256) # split input into chunks to save VRAM (shorter -> slower)

# pipeline.generate(prompt, token_count=200, args=args, callback=my_print)
# print('\n')

# # out, state = model.forward([187, 510, 1563, 310, 247], None)
# # print(out.detach().cpu().numpy())                   # get logits
# # out, state = model.forward([187, 510], None)
# # out, state = model.forward([1563], state)           # RNN has state (use deepcopy to clone states)
# # out, state = model.forward([310, 247], state)
# # print(out.detach().cpu().numpy())                   # same result as above
# # print('\n')