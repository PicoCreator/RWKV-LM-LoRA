########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################

import gc, math, os
from random import randint
from typing import List, Optional

import numpy as np
import torch
# torch._C._jit_set_profiling_executor(True)
# torch._C._jit_set_profiling_mode(True)
import torch.nn as nn
from torch.nn import functional as F

import lightning as L
from lightning.pytorch.utilities import rank_zero_info, rank_zero_only
from lightning.pytorch.strategies import DeepSpeedStrategy

import deepspeed
from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam
import deepspeed.runtime.lr_schedules
import wandb

from torch.utils.cpp_extension import load

# Script dir for various files
SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
CUDA_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "../cuda"))

########################################################################################################
# JIT / torch compile special handling
########################################################################################################

# Currently the features we need for torch compile, is avaliable only in
# 2.1 nightly build (and is expected to be in 2.1 official release)
#
# We only enable torch compile, if we either the user has explicitly
# set it, or we detect that we are running on a 2.1+ build automatically
from packaging import version
def is_torch_version_above(required_version):
    torch_version = version.parse(torch.__version__.split('+')[0])
    return torch_version >= version.parse(required_version)
IS_TORCH_2_1 = is_torch_version_above("2.0.9999")

# Get the JIT / torch compile option flags from the environment
RWKV_JIT_ON        = os.getenv("RWKV_JIT_ON", "1").lower() in ("1", "true", "yes")
RWKV_TORCH_COMPILE = os.getenv("RWKV_TORCH_COMPILE", f"{IS_TORCH_2_1}").lower() in ("1", "true", "yes")
RWKV_TORCH_RUN_MODE = None

# We enable JITMod*/Function when supporting torch.jit
# We use TorchCompile* when supporting torch compile
# based on the current runtime settings
if RWKV_TORCH_COMPILE:
    RWKV_TORCH_RUN_MODE = "torch-compile"

    JITModClass  = nn.Module
    JITModMethod = lambda x: x
    JITFunction  = lambda x: x

    # PS: i have tried mode="max-autotune", and mode="reduce-overhead", however they crash
    #     for now (8th July 2023). I may introduce them in the future once they are stable
    #
    #     Additionally, torch.compile has issues with the pytorch.lightning module directly
    # ---

    # We generally have 2 major options, either we use torch.compile
    # onto the key top level functions (train, val, test, predict, etc)
    # and let the compiler handle all the decision making on how to optimize
    #
    # However this was found to basically just match JIT level of performance exactly
    # ---
    # TCompileMax          = lambda x: x
    # TCompileBaseline     = lambda x: torch.compile(x, fullgraph=False)

    # Alternatively, we can perform a much more aggressive optimization on critical functions
    # that we know are compatible with torch.compile(fullgraph=True) - which provides the highest
    # level of optimization possible with torch.compile
    # ---
    TCompileMax        = lambda x: torch.compile(x, fullgraph=True)
    TCompileBaseline   = lambda x: x

    # ---
    # Because torch.compile is expected to change overtime, the two options should 
    # be tested every now and then, for any performance changes
    #
    # and we should switch over to the broaded automated approach if its "faster"
    # ---

    # Used to wrap functions which are **not** torch.compile compatible
    TCompileDisable    = torch._dynamo.disable

    # The following are known warnings in the nightly build, that can be safely ignored for stable release
    #
    # `torch._inductor.utils: [WARNING] DeviceCopy in input program` 
    # https://discuss.pytorch.org/t/what-can-cause-warning-devicecopy-in-input-program/175566

elif RWKV_JIT_ON:
    RWKV_TORCH_RUN_MODE = "torch-jit"
    JITModClass  = torch.jit.ScriptModule
    JITModMethod = torch.jit.script_method
    JITFunction  = torch.jit.script

    TCompileMax        = lambda x: x
    TCompileBaseline   = lambda x: x
    TCompileDisable    = lambda x: x
else:
    RWKV_TORCH_RUN_MODE = "torch-native"
    JITModClass  = nn.Module
    JITModMethod = lambda x: x
    JITFunction  = lambda x: x

    TCompileMax        = lambda x: x
    TCompileBaseline   = lambda x: x
    TCompileDisable    = lambda x: x

print(f"[RWKV.model] Running RWKV model using '{RWKV_TORCH_RUN_MODE}' with torch '{torch.__version__}'")

# ---
# Isolating out known operations that **does not work** with torch.compile
# and wrapping them within a torch._dynamo.disable, this is required to get
# the baseline torc.compile to work
# ---

@TCompileDisable
def deepspeed_checkpoint(*args, **kwargs):
    return deepspeed.checkpointing.checkpoint(*args, **kwargs)

@TCompileDisable
def wkv_op(time_decay, time_first, k, v, wkv_state):
    return torch.ops.rwkv.wkv(time_decay, time_first, k, v, wkv_state)

########################################################################################################
# RWKV: State Blocks
########################################################################################################

class TimeMixState:

    def __init__(self, shift_state: torch.Tensor, wkv_state: torch.Tensor):
        self.shift_state = shift_state
        self.wkv_state = wkv_state


class ChannelMixState:

    def __init__(self, shift_state: torch.Tensor):
        self.shift_state = shift_state


class BlockState:

    def __init__(self, time_mix_states: List[TimeMixState],
                 channel_mix_state: ChannelMixState):
        self.time_mix_states = time_mix_states
        self.channel_mix_state = channel_mix_state


class BlockStateList:

    def __init__(self, att_shift_channel_states, ffn_shift_states, wkv_shift_channel_states, att_channels):
        self.wkv_shift_channel_states = wkv_shift_channel_states
        self.att_shift_channel_states = att_shift_channel_states
        self.ffn_shift_states = ffn_shift_states
        self.att_channels = att_channels
    
    # @ TCompileMax (no difference)
    @staticmethod
    def create(N, B, C, device, dtype, att_channels):
        result = BlockStateList.empty(N, B, C, device, dtype, att_channels)
        result.wkv_states[:] = 0
        result.wkv_states[:, :, :, -1] = -1e38

        result.att_shift_channel_states[:] = 0
        
        result.ffn_shift_states[:] = 0

        return result

    # @ TCompileMax (no difference)
    @staticmethod
    def empty(N, B, C, device, dtype, att_channels):
        wkv_shift_channel_states = [torch.empty((N, B, C, 3),
                                 device=device,
                                 dtype=torch.float)] * att_channels
        
        # HOT FIX 2**12, 12 should = layer count
        att_shift_channel_states = [torch.empty((N, B, 2**12, C), device=device, dtype=dtype)] * att_channels
        ffn_shift_states = torch.empty((N, B, 1, C), device=device, dtype=dtype)
        return BlockStateList(att_shift_channel_states, ffn_shift_states, wkv_shift_channel_states)

    def __getitem__(self, layer: int):
        return BlockState(
            TimeMixState(self.att_shift_channel_states[layer], self.wkv_shift_channel_states[layer]),
            ChannelMixState(self.ffn_shift_states[layer]))

    def __setitem__(self, layer: int, state: BlockState, channel_id: int):
        # HOT FIX 2**12, 12 should = layer count
        # print("[__setitem__] layer", layer)
        # print("[__setitem__] state.time_mix_state.shift_state", state.time_mix_state.shift_state.shape)
        # print("[__setitem__] self.att_shift_states[layer]", self.att_shift_states[layer].shape)

        self.att_shift_channel_states[layer][channel_id] = state.time_mix_state[channel_id].shift_state[:,-(2**12):,:]
        self.wkv_shift_channel_states[layer][channel_id] = state.time_mix_state[channel_id].wkv_state
        self.ffn_shift_states[layer] = state.channel_mix_state.shift_state

########################################################################################################
# RWKV: RWKV Time-mix + RWKV Channel-mix
########################################################################################################

class RWKV_TimeMix(JITModClass):

    def __init__(self, layer_id, n_layer, n_embd, dim_att):
        super().__init__()

        with torch.no_grad():  # fancy init
            ratio_0_to_1 = layer_id / (n_layer - 1)  # 0 to 1
            ratio_1_to_almost0 = 1.0 - (layer_id / n_layer)  # 1 to ~0
            ddd = torch.ones(1, 1, n_embd)
            for i in range(n_embd):
                ddd[0, 0, i] = i / n_embd

            # fancy time_decay
            decay_speed = torch.ones(dim_att)
            for h in range(dim_att):
                decay_speed[h] = -5 + 8 * (h /
                                           (dim_att - 1))**(0.7 +
                                                            1.3 * ratio_0_to_1)
            self.time_decay = nn.Parameter(decay_speed)
            # print(layer_id, self.time_decay.flatten()[:3].cpu().numpy(), '...', self.time_decay.flatten()[-3:].cpu().numpy())

            # fancy time_first
            zigzag = torch.tensor([(i + 1) % 3 - 1
                                   for i in range(dim_att)]) * 0.5
            self.time_first = nn.Parameter(
                torch.ones(dim_att) * math.log(0.3) + zigzag)

            # fancy time_mix
            self.time_mix_k = nn.Parameter(torch.pow(ddd, ratio_1_to_almost0))
            self.time_mix_v = nn.Parameter(
                torch.pow(ddd, ratio_1_to_almost0) + 0.3 * ratio_0_to_1)
            self.time_mix_r = nn.Parameter(
                torch.pow(ddd, 0.5 * ratio_1_to_almost0))

        self.key = nn.Linear(n_embd, dim_att, bias=False)
        self.value = nn.Linear(n_embd, dim_att, bias=False)
        self.receptance = nn.Linear(n_embd, dim_att, bias=False)
        self.output = nn.Linear(dim_att, n_embd, bias=False)

        # shiftamount = 1 (Old version)

        shiftamount = pow(2,layer_id)
        if(shiftamount > 2048):
            shiftamount = 1
        self.time_shift = nn.ZeroPad2d((0, 0, shiftamount, -shiftamount))

    @JITModMethod
    @TCompileMax
    def _forward_kvsr(self, x, last_state: TimeMixState):

        # Print the various shapes for debugging
        # print("")
        # print("x shape: ", x.shape) # eg. [1, 2909, 2560]
        # print("last_state.wkv_state: ", last_state.wkv_state.shape) # eg. [1, 2560, 3]
        # print("last_state.shift_state: ", last_state.shift_state.shape) # eg. [4096, 1, 2560]
        # print("last_state.shift_state.unsqueeze(0): ", last_state.shift_state.unsqueeze(0).shape) # eg. [1, 1, 2560]
        
        xxx = torch.concat((last_state.shift_state, x), dim=1)
        
        xxxx = self.time_shift(xxx)
        xx = xxxx[:, -x.shape[1]:, :]
        # xx = xxxx.view(x.shape[0], x.shape[1], x.shape[2])
        # print( "xxxx shape: ", xxxx.shape) # eg. [1, 2582, 2560]
        # print( "xxx shape: ", xxx.shape) # eg. [1, 2582, 2560]
        # print( "xx.shape: ", xx.shape) # eg. [1, 2582, 2560]
        # print( "x.shape: ", x.shape) # eg. [1, 2488, 2560]
        # print( "self.time_mix_k.shape: ", self.time_mix_k.shape)

        # last_state.shift_state
        # xx

        xk = x * self.time_mix_k + xx * (1 - self.time_mix_k)
        xv = x * self.time_mix_v + xx * (1 - self.time_mix_v)
        xr = x * self.time_mix_r + xx * (1 - self.time_mix_r)
        

        k = self.key(xk)
        v = self.value(xv)
        r = self.receptance(xr)

        sr = torch.sigmoid(r)

        # Enforce bf16 type for kv, as this can be mis init
        # when being called directly via inference
        if k.dtype != torch.bfloat16:
            k = k.to(torch.bfloat16)
        if v.dtype != torch.bfloat16:
            v = v.to(torch.bfloat16)

        return k, v, sr, xxx

    @JITModMethod
    @TCompileMax
    def _forward_out(self, sr, y, x_l, new_wkv_state):
        return self.output(sr * y), TimeMixState(x_l, new_wkv_state)

    @JITModMethod
    @TCompileBaseline
    def forward(self, x, last_state: TimeMixState):
        # Enforce bf16 for self.time_first
        # as this can be mis init when being called directly via inference
        if self.time_first.dtype != torch.bfloat16:
            self.time_first = self.time_first.to(torch.bfloat16)

        # Perform the WKV op via cuda code
        k, v, sr, xxx = self._forward_kvsr(x, last_state)
        y, new_wkv_state = wkv_op(self.time_decay, self.time_first,
                                  k, v, last_state.wkv_state)
        
        return self._forward_out(sr, y, xxx, new_wkv_state)


########################################################################################################


class RWKV_ChannelMix(JITModClass):

    def __init__(self, layer_id, n_layer, n_embd, dim_ffn):
        super().__init__()

        with torch.no_grad():  # fancy init of time_mix
            ratio_1_to_almost0 = 1.0 - (layer_id / n_layer)  # 1 to ~0
            ddd = torch.ones(1, 1, n_embd)
            for i in range(n_embd):
                ddd[0, 0, i] = i / n_embd
            self.time_mix_k = nn.Parameter(torch.pow(ddd, ratio_1_to_almost0))
            self.time_mix_r = nn.Parameter(torch.pow(ddd, ratio_1_to_almost0))

        self.key = nn.Linear(n_embd, dim_ffn, bias=False)
        self.receptance = nn.Linear(n_embd, n_embd, bias=False)
        self.value = nn.Linear(dim_ffn, n_embd, bias=False)

    @JITModMethod
    @TCompileMax
    def forward(self, x, last_state: ChannelMixState):
        xx = torch.concat((last_state.shift_state, x[:, :-1]),
                          dim=1)
        
        # print("[CM] x shape: ", x.shape)
        # print("[CM] last_state.shift_state shape: ", last_state.shift_state.shape)
        # print("[CM] xx shape: ", xx.shape)
        # print("[CM] self.time_mix_k shape: ", self.time_mix_k.shape)
        # print("[CM] self.time_mix_r shape: ", self.time_mix_r.shape)

        xk = x * self.time_mix_k + xx * (1 - self.time_mix_k)
        xr = x * self.time_mix_r + xx * (1 - self.time_mix_r)
        k = self.key(xk)
        k = torch.square(torch.relu(k))
        kv = self.value(k)
        return (torch.sigmoid(self.receptance(xr)) * kv,
                ChannelMixState(x[:, -1]))


########################################################################################################
# The RWKV Model blocks
########################################################################################################

class Block(nn.Module):

    def __init__(self, layer_id, n_layer, n_embd, dim_att, dim_ffn, att_channels):
        super().__init__()
        self.layer_id = layer_id

        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

        if self.layer_id == 0:
            self.ln0 = nn.LayerNorm(n_embd)

        self.att_channels = att_channels
        
        self.att = [ RWKV_TimeMix(layer_id, n_layer, n_embd, dim_att) for i in range(self.att_channels) ]

        self.ffn = RWKV_ChannelMix(layer_id, n_layer, n_embd, dim_ffn)

    def forward(self, x, last_state: BlockState):
        if self.layer_id == 0:
            x = self.ln0(x)
            
        att_outs = []
        att_states = {}
        
        x_ln1 = self.ln1(x)
        for i in range(self.att_channels):
            att_out, att_state = self.att[i](
                x_ln1,
                last_state.time_mix_states[self.layer_id][i],
            )
            att_outs.append(att_out)
            
            # making sure the order doesn't somehow get messed up in the state variable (i don't feel like testing this) - nathan
            att_states[i] = att_state
            
        # matrix addition of all attention outputs
        att_out = torch.sum(torch.stack(att_outs), dim=0)
        
        x = x + att_out
        ffn_out, ffn_state = self.ffn(
            self.ln2(x),
            last_state.channel_mix_state,
        )
        x = x + ffn_out
        return x, BlockState(att_states, ffn_state)


class L2Wrap(torch.autograd.Function):

    @staticmethod
    def forward(ctx, loss, y, token_amount, currentMask):
        # Currently (8th July 2023), save_for_backward, causes an issue with
        # pytorch.compile (see: https://github.com/pytorch/pytorch/blob/e600505e3209eaf539e8bc99870ea55236cefbf5/torch/_dynamo/variables/higher_order_ops.py#L735)
        # 
        # Due to L2Wrap being a major hotspot, we should monitor this for future support.
        # so that once its resolved, we can include the L2Wrap step in the torch.compile path
        #
        # See also:
        # - checkpointed_step
        ctx.save_for_backward(y)
        ctx.token_amount = token_amount
        ctx.currentMask = currentMask
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        y, = ctx.saved_tensors
        token_amount = ctx.token_amount
        # to encourage the logits to be close to 0
        factor = 1e-4 / token_amount
        maxx, ids = torch.max(y, -1, keepdim=True)
        gy = torch.zeros_like(y)
        gy.scatter_(-1, ids, maxx * factor)
        gy = gy * ctx.currentMask[:, None][None, :]
        return (grad_output, gy, None, None)

########################################################################################################
# Static optimized functions
########################################################################################################

# @ TCompileMax (no speed improvement)
# def F_cross_entropy_reduction_none_optimized(logits, targets):
#     return F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), reduction="none")

########################################################################################################
# Core RWKV module
########################################################################################################
class RWKV(L.LightningModule):

    def __init__(self,
                 ctx_len: int,
                 n_embd: int,
                 n_layer: int,
                 vocab_size: int,
                 # Model file path to load from
                 load_model: Optional[str] = None,
                 # Context length schedule
                 ctx_len_cutoffs: List[int] = [],
                 ctx_len_warmup_steps: List[int] = [],
                 # Alternative to lr_init / lr_final
                 # that is multiplied by the gradient_accumulation_steps
                 # to get the actual learning rate
                 target_lr_init: float = -1.0,
                 target_lr_final: float = -1.0,
                 # Learning rate schedule
                 # use only target_lr_init / lr_init
                 # to configure a constant learning rate
                 lr_init: float = -1.0,
                 lr_final: float = -1.0,
                 lr_period: int = -1,
                 lr_period_type: str = 'epoch',
                 # Adam optimizer settings
                 beta1: float = 0.9,
                 beta2: float = 0.99,
                 adam_eps: float = 1.0e-08,
                 weight_decay: float = 0.01,
                 warmup_steps: int = -1,
                 # Backprop settings
                 grad_cp: bool = True,
                 bptt_learning: bool = True,
                 bptt_learning_range: int = -1,
                 bptt_truncated_learning: bool = False,
                 layerwise_lr: bool = True,
                 dim_att: Optional[int] = None,
                 dim_ffn: Optional[int] = None,
                 substep_cuda_cache_clear: bool = False,
                 substep_logging: bool = False,
                 torch_set_float32_matmul_precision:str = 'high',
                 time_shift_channels: int = 4
                 ):

        # Lets save everything in one shot
        # (this is used for wandb logging)
        self.setup_args = locals()
        del self.setup_args["self"]
        del self.setup_args["__class__"]

        # Setup the model
        super().__init__()

        # Save the various other params for later
        self.ctx_len = ctx_len
        self.ctx_len_cutoffs = ctx_len_cutoffs
        self.ctx_len_warmup_steps = ctx_len_warmup_steps
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.layerwise_lr = layerwise_lr
        self.grad_cp = grad_cp
        self.target_lr_init = target_lr_init
        self.target_lr_final = target_lr_final
        self.lr_init = lr_init
        self.lr_final = lr_final
        self.lr_period = lr_period
        self.lr_period_type = lr_period_type
        self.warmup_steps = warmup_steps
        self.beta1 = beta1
        self.beta2 = beta2
        self.weight_decay = weight_decay
        self.adam_eps = adam_eps
        self.bptt_learning = bptt_learning
        self.bptt_learning_range = bptt_learning_range
        self.bptt_truncated_learning = bptt_truncated_learning
        self.substep_cuda_cache_clear = substep_cuda_cache_clear
        self.substep_logging = substep_logging

        dim_att = dim_att or n_embd
        dim_ffn = dim_ffn or n_embd * 4

        if torch_set_float32_matmul_precision is not None:
            torch.set_float32_matmul_precision(torch_set_float32_matmul_precision)

        self.emb = nn.Embedding(vocab_size, n_embd)

        load(name=f"wkv_{self.ctx_len}_bf16",
             sources=[
                os.path.join(CUDA_DIR, "wkv_op_bf16.cpp"),
                os.path.join(CUDA_DIR, "wkv_cuda_bf16.cu")
            ],
             verbose=True,
             extra_cflags=["-std=c++17", "-O3", f"-DTmax={self.ctx_len}"],
             extra_cuda_cflags=[
                 "-t 4", "-std=c++17", "-res-usage", "--maxrregcount 60",
                 "--use_fast_math", "-O3", "-Xptxas -O3",
                 "--extra-device-vectorization", f"-DTmax={self.ctx_len}"
             ],
             is_python_module=False)
        
        self.time_shift_channels = time_shift_channels

        self.blocks = nn.ModuleList([
            Block(i, n_layer, n_embd, dim_att, dim_ffn, self.time_shift_channels) for i in range(n_layer)
        ])

        self.ln_out = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size, bias=False)

        # self.load_state_dict(torch.load(load_model, map_location='cpu'))

    def configure_optimizers(self):
        if self.bptt_learning == False:
            if self.deepspeed_stage >= 2 or self.deepspeed_offload:
                print(f"[WARNING]: it is highly recommended to enable bptt_learning when used to deepspeed 2/3/offloading, otherwise an exception will occur when training with dataset records, larger then the configured context length ({self.ctx_len})")
        else:
            if self.trainer.num_devices > 1:
                if self.bptt_learning_range <= 0:
                    print("[WARNING]: unlimited bptt_learning_range across multiple GPU's has a performance penalty with datasets of mixed sizes due to its constant need to keep all GPU's in sync (consider using bptt_learning_range=1 instead)")
                if self.bptt_learning_range > 1:
                    # Temporary error, till better sync logic is done for mixed document sizes
                    # (lazy to support this right now, since i have no idea if anyone has a use for it)
                    raise NotImplementedError("bptt_learning_range > 1 is not supported yet")
        
        # Get the learning rate used for the optimizer
        lr_init = self.lr_init
        lr_final = self.lr_final
        target_lr_init = self.target_lr_init
        target_lr_final = self.target_lr_final

        assert not (target_lr_init <= 0.0 and lr_init <= 0.0), f"Either 'target_lr_init' ({target_lr_init})  or 'lr_init' ({lr_init}) must be specified"
        assert not (target_lr_init > 0.0 and lr_init >= 0.0), f"Use either 'target_lr_init'  ({target_lr_init}) or 'lr_init' ({lr_init})"
        
        if target_lr_init > 0.0:
            assert self.trainer.target_batch_size > 0, "'target_batch_size' must be specified in the trainer, when using 'target_lr_init'"
            lr_init = target_lr_init * self.trainer.accumulate_grad_batches / self.trainer.target_batch_size
        if target_lr_final > 0.0:
            assert self.trainer.target_batch_size > 0, "'target_batch_size' must be specified in the trainer, when using 'target_lr_final'"
            lr_final = target_lr_final * self.trainer.accumulate_grad_batches / self.trainer.target_batch_size

        # If the final learning rate is not specified, use the initial learning rate
        if lr_final < 0:
            lr_final = self.lr_init

        # Log the learning rate, and various other parameters
        if self.trainer.local_rank == 0:
            start_lr_e = "{:.3e}".format(lr_init)
            lr_final_e = "{:.3e}".format(lr_final)
            print(f"\n[RWKV.model] Configuring optimizer with\n"+
                  f"    - lr_init:  {start_lr_e} ({lr_init})\n"+
                  f"    - lr_final: {lr_final_e} ({lr_final})\n")

            # Get the setup args
            model_args = dict(self.setup_args)
            model_args["__lr_init"] = lr_init
            model_args["__lr_final"] = lr_final

            # Update WANDB
            if wandb.run is not None:
                wandb.config.update({ "model": model_args })

        # Setup layerwise learning rate
        if self.layerwise_lr:
            lr_1x = set()
            lr_2x = set()
            lr_3x = set()
            for n, p in self.named_parameters():
                if "time_mix" in n:
                    lr_1x.add(n)
                elif "time_decay" in n:
                    lr_2x.add(n)
                elif "time_first" in n:
                    lr_3x.add(n)
                else:
                    lr_1x.add(n)
            lr_1x = sorted(list(lr_1x))
            lr_2x = sorted(list(lr_2x))
            lr_3x = sorted(list(lr_3x))
            # print('1x', lr_1x)
            # print('2x', lr_2x)
            # print('3x', lr_3x)
            param_dict = {n: p for n, p in self.named_parameters()}
            optim_groups = [
                {
                    "params": [param_dict[n] for n in lr_1x],
                    "weight_decay": 0.0,
                    "lr": 1.0 * lr_init
                },
                {
                    "params": [param_dict[n] for n in lr_2x],
                    "weight_decay": 0.0,
                    "lr": 2.0 * lr_init
                },
                {
                    "params": [param_dict[n] for n in lr_3x],
                    "weight_decay": 0.0,
                    "lr": 3.0 * lr_init
                },
            ]
        else:
            optim_groups = [
                {
                    "params": [p for n, p in self.named_parameters()],
                    "weight_decay": 0.0
                },
            ]

        # Setup the adam optimizers
        if self.deepspeed_offload:
            optimizer = DeepSpeedCPUAdam(optim_groups,
                                         lr=lr_init,
                                         betas=(self.beta1, self.beta2),
                                         eps=self.adam_eps,
                                         bias_correction=True,
                                         adamw_mode=False,
                                         weight_decay=self.weight_decay,
                                         amsgrad=False)
        else:
            optimizer = FusedAdam(optim_groups,
                                  lr=lr_init,
                                  betas=(self.beta1, self.beta2),
                                  eps=self.adam_eps,
                                  bias_correction=True,
                                  adam_w_mode=False,
                                  weight_decay=self.weight_decay,
                                  amsgrad=False)
            
        # Throw if wramup_steps and lr_period are both set (not supported)
        if self.warmup_steps > 0 and self.lr_period > 0:
            raise ValueError(
                "Use either warmup_steps or lr_period, not both.")

        if self.warmup_steps > 0:
            lr_scheduler = deepspeed.runtime.lr_schedules.WarmupLR(
                optimizer,
                warmup_min_lr=0.2 * self.lr_init,
                warmup_max_lr=self.lr_init,
                warmup_num_steps=self.warmup_steps,
                warmup_type='linear')

            return {
                'optimizer': optimizer,
                'lr_scheduler': lr_scheduler,
            }

        else:
            # Skip the lr_scheduler process if lr_init and lr_final are the same
            if lr_init == lr_final:
                return optimizer

            # The total number of steps to perform training rate decay with
            lr_total_step = 0

            # Handle lr_period -1 default behaviour of using the max_step / max_epoch
            if self.lr_period == -1:
                # Get trainer max_step / max_epoch
                trainer_max_step = self.trainer.max_steps
                trainer_max_epoch = self.trainer.max_epochs
                if trainer_max_step > 0:
                    lr_total_step = trainer_max_step
                elif trainer_max_epoch > 0:
                    lr_total_step = trainer_max_epoch * self.num_step_per_epoch()
                else :
                    print("Warning: max_step/max_epoch not set, we would be performing lr_init to lr_final shift assuming 10 epoch")
                    lr_total_step = 10 * self.num_step_per_epoch()
            else:
                # Calculate lr_total_step based on lr_period
                if self.lr_period_type == "step":
                    lr_total_step = self.lr_period
                elif self.lr_period_type == "epoch":
                    lr_total_step = self.lr_period * self.num_step_per_epoch()
                else:
                    raise ValueError(f"lr_period_type {self.lr_period_type} not supported.")

            # Lets initialize the lr_scheduler
            lr_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=1.0,
                end_factor= lr_final / lr_init,
                total_iters=lr_total_step
            )

            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    "scheduler": lr_scheduler,
                    "interval": "step",
                    "frequency": 1,
                },
            }
                
    
    # We have to compute the number of steps per epoch ourselves
    # as this value is not provided directly by pytorch lightning
    # https://github.com/Lightning-AI/lightning/issues/5449#issuecomment-1501597319
    def num_step_per_epoch(self) -> int:
        # Estimated number of steps in total, added as the following
        # https://github.com/Lightning-AI/lightning/pull/11599
        #
        # This MUST be called before len(self.trainer.train_loader)
        # otherwise there is a bug in which the train_dataloader is not
        # fully initialized, which seems to be resolved by computing the 
        # self.trainer.estimated_stepping_batches
        estimated_stepping_batches = self.trainer.estimated_stepping_batches

        # Get the number of epochs, 
        # use estimated_stepping_batches if max_epochs is set
        max_epochs = self.trainer.max_epochs
        if max_epochs > 0:
            return estimated_stepping_batches // max_epochs

        # Get the train_dataloader
        train_dataloader = self.trainer.train_dataloader
        if train_dataloader is None:
            train_dataloader = self.trainer.fit_loop._data_source.dataloader()

        # Max epoch is not set, use the train_dataloader
        dataset_size = len(train_dataloader)

        num_devices = max(1, self.trainer.num_devices)
        num_steps = dataset_size // (self.trainer.accumulate_grad_batches * num_devices)
        return num_steps
    
    @property
    def deepspeed_offload(self) -> bool:
        strategy = self.trainer.strategy
        if isinstance(strategy, DeepSpeedStrategy):
            cfg = strategy.config["zero_optimization"]
            return "offload_optimizer" in cfg or "offload_parameters" in cfg
        return False
    
    @property
    def deepspeed_stage(self) -> int:
        strategy = self.trainer.strategy
        if isinstance(strategy, DeepSpeedStrategy):
            cfg = strategy.config["zero_optimization"]
            return "stage" in cfg
        return -1

    @TCompileBaseline
    def forward(self, idx: torch.Tensor, last_att_shift_channel_states: List[torch.Tensor], last_ffn_shift_states: torch.Tensor,
                last_wkv_states: torch.Tensor):
        B, T = idx.size()
        assert T <= self.ctx_len, "Cannot forward, model ctx_len is exhausted."

        x = self.emb(idx)

        new_states = BlockStateList.empty(self.n_layer, B, self.n_embd,
                                          x.device, x.dtype, self.time_shift_channels)
        
        if last_att_shift_channel_states is None:
            cur_bs_list = BlockStateList.empty(
                self.n_layer, B,
                self.n_embd,
                x.device,
                x.dtype,
                self.time_shift_channels
            )
        else:
            cur_bs_list = BlockStateList(last_att_shift_channel_states,last_ffn_shift_states, last_wkv_states)

        # Avoid using the zip operation, as torch.compile throws an exception on it
        # with `zip not reconized as a valid function`
        # ---
        # for i, (block, last_state) in enumerate(
        #         zip(self.blocks,
        #             BlockStateList(last_shift_states, last_wkv_states))):
        # ---
        for i in range(len(self.blocks)):
            block = self.blocks[i]
            last_state = cur_bs_list[i]
            if self.grad_cp:
                x, new_state = deepspeed_checkpoint(
                    block, x, last_state)
            else:
                x, new_state = block(x, last_state)
            new_states[i] = new_state

        x = self.ln_out(x)

        x = self.head(x)

        return x, new_states.att_shift_channel_states, new_states.ffn_shift_states, new_states.wkv_shift_channel_states

    #
    # Custom overwrite of manual_backwards operation, to skip the "manual_backwards"
    # safety check, so we can perform manual backward operation step, while using
    # the default trainer loop. This is modified from the original code found here:
    # https://github.com/Lightning-AI/lightning/blob/37c244f94be365496def82870b22c2faf0ab889e/src/lightning/pytorch/core/module.py#L999
    #
    # ---
    # 
    # This allow us to avoid disabling the "automatic_optimization" flag
    #
    # Which would have been required to do "segmented learning", or "Backpropagation Through Time"
    # where we would need to implement manual optimization as per
    # https://lightning.ai/docs/pytorch/stable/model/manual_optimization.html
    #
    # Otherwise an error will be thrown if we call `self.manual_backward`
    #
    # However this would mean that we would need to do a full reimplementation
    # of several features that were handled by the automatic optimization.
    # - accumulate_grad_batches
    # - gradient_clip_val
    # - logging behaviour
    # - distributed training co-ordination
    # - (And probably other features that I am not aware of)
    #
    # So this is a hacky work around, to avoid reimplementing all of the above.
    # 
    # From the current code implementatiion, it seem like this is blocked only by 
    # automatic_optimization flag - and has no adverse side effect otherwise
    # https://lightning.ai/docs/pytorch/stable/_modules/lightning/pytorch/core/module.html#LightningModule.manual_backward
    #
    # If anyone have a better idea, let me know
    # (have experimented with, reimplementing the above, but it is not trivial, unfortunately)
    #
    def manual_backward(self, loss: torch.Tensor, *args, **kwargs):
        if self._fabric:
            self._fabric.backward(loss, *args, **kwargs)
        else:
            # self._verify_is_manual_optimization("manual_backward")
            self.trainer.strategy.backward(loss, None, *args, **kwargs)

    #
    # Main compute_loss function, this is called by the trainer loop
    #
    def compute_loss(self, batch, batch_idx, is_training_run: bool):
        seq = batch['input_ids']
        assert isinstance(seq, torch.Tensor) and seq.ndim == 2
        seq_mask = batch['attention_mask']

        # Check if attent mask is set, if not initialize it
        if seq_mask is None or seq_mask.ndim != 2:
            seq_mask = torch.ones_like(seq[:, 1:])

        # Perform cutoff for training run
        if is_training_run:
            prev_step = 0

            # Avoid using the zip operation, as torch.compile throws an exception on it
            # with `zip not reconized as a valid function`
            # ---
            # for step, len_cut in zip(self.ctx_len_warmup_steps,
            #                          self.ctx_len_cutoffs):
            # ---
            for i in range(min(len(self.ctx_len_warmup_steps), len(self.ctx_len_cutoffs))):
                step = self.ctx_len_warmup_steps[i]
                len_cut = self.ctx_len_cutoffs[i]

                if prev_step <= self.global_step < step and len_cut < seq.shape[
                        1] - 1:
                    pos = randint(0, seq.shape[1] - len_cut - 1)

                    # Original
                    # seq = seq[:, pos:pos + len_cut + 1]

                    # Changed to use masking for prefix cutoff (i do not know if this makes sense)
                    seq = seq[:, :pos + len_cut + 1]
                    seq_mask = seq_mask[:, :pos + len_cut + 1]
                    # Set the attention mask to 0 for the skipped tokens
                    seq_mask[:, :pos] = 0
                    break
                prev_step = step
                
        do_bptt_learning = self.bptt_learning and is_training_run
        idx, targets = seq[:, :-1], seq[:, 1:]

        B, T = idx.shape
        C = self.n_embd
        total_mask_sum = torch.sum(seq_mask)

        # If total_mask_sum, we skip, as there is no tokens of value to learn from anyway
        if total_mask_sum == 0:
            return 0
        
        def checkpointed_step(idx, targets, mask, prev_loss, last_att_shift_states, last_ffn_shift_states,
                              last_wkv_states, prev_steps):
            logits, new_att_shift_states, new_ffn_shift_states, new_wkv_states = self(
                idx, last_att_shift_states, last_ffn_shift_states, last_wkv_states)
            
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                                    targets.view(-1),
                                    reduction="none")

            submask = mask.view(-1)[:loss.shape[0]]
            submask_sum = torch.sum(submask)
            loss = torch.sum(loss * submask) / total_mask_sum  

            loss = L2Wrap.apply(loss, logits, total_mask_sum, submask)
            new_steps = prev_steps + submask_sum
            new_loss = prev_loss + loss
            
            return new_loss, new_att_shift_states, new_ffn_shift_states, new_wkv_states, new_steps

        total_loss = torch.tensor(
            0, dtype=self.emb.weight.dtype).requires_grad_()
        steps = 0
        states = BlockStateList.create(self.n_layer, B, C, seq.device,
                                       self.emb.weight.dtype, self.time_shift_channels)
        segment_count = math.ceil(T / self.ctx_len)

        #
        # BPTT learning, we split the sequence into segments
        # and perform a backward pass for each segment, on its own.
        #
        # Allowing us to perform backpropagation across context sizes much larger
        # then what is supported by the current GPU memory.
        #
        # This reduces the need for the checkpointing process, and mitigate
        # a known error where multiple backwards pass throws an exception.
        #
        # While not mathematically equivalent to full context size learning,
        # it makes "infctx" size training possible with deepspeed 2/3
        #
        # ---
        # 
        # See the following, for more details on "Gradient computed twice" error:
        # https://github.com/microsoft/DeepSpeed/issues/988#issuecomment-1549417269
        #
        # Other possibly related issues on the topic:
        # https://github.com/microsoft/DeepSpeed/pull/677
        # https://github.com/EleutherAI/gpt-neox/issues/62#issuecomment-766366413
        #
        if do_bptt_learning:

            gradient_accumulation_steps = max(1, self.trainer.accumulate_grad_batches)
            optimizer = self.optimizers()
            cur_device = self.device
            
            # We use the average segment size, instead of ctx length size.
            # this helps ensure that the segment cutoffs do not make the last segment too small.
            # (eg, the last chunk having only 1 token)
            #
            # it also helps ensure the segment cutoff points are more varied, across mixed dataset sizes
            # and avoid potentially undesired training behaviour at fixed cutoff points
            # (this only applies for segmented learning)
            segment_size = min(math.ceil(T / segment_count), self.ctx_len)

            # We compute when we start the segmented learning process
            if self.bptt_learning_range > 0:
                first_learning_segment = segment_count - self.bptt_learning_range;
            else:
                first_learning_segment = 0;

            # Dummy 2D tenros of shape [1,1], are used to do "dummy checkpoint/forward/backprop" to keep everything in sync
            dummy_2d_zero = torch.tensor([[0]], dtype=torch.long, device=cur_device)

            # Get the max segment count across all GPUs, in the current batch, which is used to keep all devices are in sync
            # Once a thread has completed all its segments, it will do dummy checkpoint/forward/backprop with one token,
            # and stay in sync with the thread that are still working on their segments
            #
            # This is used to work around an undocumented behaviour for either lightning / deepspeed loss.backward multi-gpu handling
            # where the `self.manual_backward()` / `loss.backward()` call will block / freeze / hang when being too "out of sync"
            #
            # This can be viewed as a form of `fabric.barrier()` which is invoked implicitly by each `self.manual_backward()` call
            # except that it isn't exactly a `fabric.barrier()` - because it does not block immediately and instead blocks in 
            # the next `self.manual_backward()` call if the previous ones are too far out of sync. 
            # (its confusing, but makes sense for efficency)
            #
            # Additionally because the "code line position" and params actually matter for the 'barrier' code effect,
            # we cant work around this issue by doing dummy `self.manual_backward()` calls, in another if/else branch or loop
            #
            # Combined, this makes this issue very hard to trace and debug, as it will manifest itself as randomly "freezing"
            # when out of sync during `self.manual_backward()` calls. Without the current work around put in place
            #
            # We only do this, if we are doing bptt learning on more then 1 segment, and gpu count > 1
            # otherwise we just use the segment count as it is
            if self.trainer.num_devices > 1 and (self.bptt_learning_range > 1 or self.bptt_learning_range <= 0):
                shared_segment_count = self.trainer.getFabric().all_reduce(segment_count, reduce_op="max").item()
            else:
                shared_segment_count = segment_count

            # Lets go through all the segments (including dummy ones)
            for i in range(shared_segment_count):
                # Apply state truncation, if truncated learning is enabled
                # this limits the backprop process, reduces loss learning rate, 
                # but save vram across extreamly large backpropagation steps
                if self.bptt_truncated_learning:
                    prv_att_shift_states = states.att_shift_states.clone().detach().requires_grad_(False)
                    prv_ffn_shift_states = states.ffn_shift_states.clone().detach().requires_grad_(False)
                    prv_wkv_states = states.wkv_states.clone().detach().requires_grad_(False)
                else:
                    prv_att_shift_states = states.att_shift_states
                    prv_ffn_shift_states = states.ffn_shift_states
                    prv_wkv_states = states.wkv_states
                
                # We use a dummy masked token 0, to do additional dummy checkpoint/forward/backprop when needed
                # for each additional call after the current "segment_count" max
                if i <= segment_count - 1:
                    cur_idx = idx[:, i * segment_size:(i + 1) * segment_size]
                    cur_tar = targets[:, i * segment_size:(i + 1) * segment_size]
                    cur_msk = seq_mask[:, i * segment_size:(i + 1) * segment_size]
                else:
                    cur_idx = dummy_2d_zero
                    cur_tar = dummy_2d_zero
                    cur_msk = dummy_2d_zero

                # Segmented learning, applies the forward/pass over each chunk seperately
                segment_loss, new_att_shift_states, new_ffn_shift_states, new_wkv_states, steps = checkpointed_step(
                    cur_idx,
                    cur_tar,
                    cur_msk,
                    torch.tensor(0, dtype=self.emb.weight.dtype, device=cur_device).requires_grad_(True),
                    prv_att_shift_states,
                    prv_ffn_shift_states,
                    prv_wkv_states,
                    steps,
                )
                states = BlockStateList(new_att_shift_states, new_ffn_shift_states, new_wkv_states)
                
                # Compute the backward pass for the segment
                if i >= first_learning_segment:
                    # The learning loss, should be normalized against the accumulation steps
                    # as we are bypassing the pytorch lightning normalization
                    # https://lightning.ai/docs/pytorch/2.0.4/common/lightning_module.html#backward
                    learning_loss = segment_loss / gradient_accumulation_steps

                    # Perform the backward pass accordingly
                    if i < shared_segment_count-1:
                        # Undocumented multiple backward pass support
                        # https://github.com/Lightning-AI/lightning/blob/678f642808c54e4c490caee4df5d357301c976bb/tests/trainer/optimization/test_manual_optimization.py#L251
                        self.manual_backward(learning_loss, optimizer, retain_graph=True)

                        # Accumulate without gradient, as we already did the backward pass
                        total_loss = total_loss + segment_loss.clone().detach().requires_grad_(False)
                    else:
                        # This is the last pass, we let the default pytorch lightning handle the backward pass
                        # and return the segment loss as part of the total loss
                        total_loss = total_loss + segment_loss
                else:
                    # Even if its not the segments we use for backward pass, we need to accumulate the loss
                    total_loss = total_loss + segment_loss.clone().detach().requires_grad_(False)

                # GC collect unused memory
                gc.collect()
                # torch.cuda.empty_cache()
        else:

            # Normal operations without BPTT
            segment_size = self.ctx_len
            for i in range(segment_count):
                if i < segment_count-1 and is_training_run:
                    total_loss, new_att_shift_states, new_ffn_shift_states, new_wkv_states, steps = deepspeed_checkpoint(
                        checkpointed_step,
                        idx[:, i * segment_size:(i + 1) * segment_size],
                        targets[:, i * segment_size:(i + 1) * segment_size],
                        seq_mask[:, i * segment_size:(i + 1) * segment_size],
                        total_loss,
                        states.att_shift_states,
                        states.ffn_shift_states,
                        states.wkv_states,
                        steps,
                    )
                else:
                    total_loss, new_att_shift_states, new_ffn_shift_states, new_wkv_states, steps = checkpointed_step(
                        idx[:, i * segment_size:(i + 1) * segment_size],
                        targets[:, i * segment_size:(i + 1) * segment_size],
                        seq_mask[:, i * segment_size:(i + 1) * segment_size],
                        total_loss,
                        states.att_shift_states,
                        states.ffn_shift_states,
                        states.wkv_states,
                        steps,
                    )

                states = BlockStateList(new_att_shift_states, new_ffn_shift_states, new_wkv_states)
                gc.collect()
                # torch.cuda.empty_cache()

        # Wandb logging only, if an active run exists
        if wandb.run is not None:
            global_rank = self.global_rank
            global_device_count = self.trainer.num_devices * self.trainer.num_nodes
            wandb.log({
                'substep': batch_idx * global_device_count + global_rank,
                'batchidx': batch_idx,
                'global_rank': global_rank, 
                'real_ctx_len': T, 
                'train/loss': total_loss,
                'trainer/global_step':self.global_step,
                'trainer/learning_rate': self.trainer.optimizers[0].param_groups[0]['lr']
            })

        return total_loss

    @TCompileBaseline
    def training_step(self, batch, batch_idx):
        total_loss = self.compute_loss(batch, batch_idx, True)

        self.log('train/loss', total_loss, prog_bar=True)
        # If set - forces the above train/loss log line to always be on a new line
        if self.substep_logging:
            print("")
        
        if self.substep_cuda_cache_clear:
            gc.collect()
            torch.cuda.empty_cache()

        return total_loss

    @TCompileBaseline
    def validation_step(self, batch, batch_idx):
        total_loss = self.compute_loss(batch, batch_idx, False)
        self.log('validation/loss', total_loss, prog_bar=True, sync_dist=True)
        return total_loss

########################################################################################################
# SimpleRWKV, a wrapper for RWKV that allows for simple usage of the model
########################################################################################################

# SimpleRWKV specific imports
from transformers import PreTrainedTokenizerFast

# Current script dir
SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
SCRIPT_PARENT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '../'))

# SimpleRWKV is a wrapper for RWKV that allows for simple usage of the model
#
# it is not meant to be highly performant, but rather a simple minimal way to run the RWKV trainer module
# in inference mode, and can be used to validate the model trainer code / its changes
class SimpleRWKV():

    def __init__(
            self,
            model_path: str,
            ctx_len:int = 1024,
            device:str = "cuda",
            dtype:str = "fp32",
            tokenizer = "pile",
        ):

        # Device type must be cuda, cpu type is not supported (yet?)
        if device != "cuda":
            raise NotImplementedError("Only cuda device is supported (for now)")

        # Setup the tokenizer
        if tokenizer == "pile":
            tokenizer_file = os.path.join(SCRIPT_PARENT_DIR,"20B_tokenizer.json")
            tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_file)
            vocab_size = 50277
        else:
            raise NotImplementedError("Only pile tokenizer is supported")
        self.fastTokenizer = tokenizer

        # Load the model, yes i know this is a double load 
        # but the goal of SimpleRWKV is not to optimize init, but to "just work"
        _torch_load = torch.load(model_path, map_location="cpu")

        # Get the model params
        keys = list(_torch_load.keys())

        # Get the maximum block id
        max_block_id = 0
        for x in keys:
            if 'blocks.' in x:
                block_id = int(x.split('.')[1])
                max_block_id = max(max_block_id, block_id)
        
        # Compute the layer count, embed sizes, and vocab size
        n_layer = max_block_id + 1
        n_embd = _torch_load['head.weight'].shape[1]
        vocab_size = max(_torch_load['head.weight'].shape[0], vocab_size)

        # Lets delete the _torch_load to free up memory
        del _torch_load

        # Prepare the model config with the model path, and custom torch load
        model_config = {}
        model_config["load_model"] = model_path
        model_config["n_embd"] = n_embd 
        model_config["n_layer"] = n_layer 
        model_config["vocab_size"] = vocab_size 
        model_config["ctx_len"] = ctx_len

        # This feature depends on deepspeed
        model_config["grad_cp"] = False
        # model_config["_torch_load_state"] = loaded_state

        # Save the config settings
        self.ctx_len = ctx_len
        self.device = device

        # Lets actually load the model
        self.model = RWKV(**model_config)

        # Lets map it over to the respective device type
        # and set it to run as eval/inference mode
        self.model.to(device)
        self.model.eval()

    # Encoding strings
    def encode(self, text: str):
        return self.fastTokenizer.encode(text)

    # Decoding strings
    def decode(self, tokens: list):
        return self.fastTokenizer.decode(tokens)

    # Forwarding logic, withoout torch._no_grad() context
    def _forward(
            self, tokens:list, 
            stateObj = None
        ):

        logits_arr = None
        token_len = len(tokens)

        # Get the shift/wkv state
        if stateObj is None:
            att_shift_states = None
            ffn_shift_states = None
            wkv_states = None
        else:
            att_shift_states = stateObj["att_shift_states"]
            ffn_shift_states = stateObj["ffn_shift_states"]
            wkv_states = stateObj["wkv_states"]
        
        # For each token, process the state, in batches up to ctx_len
        for i in range(0, token_len, self.ctx_len):
            # Get the tokens for this batch
            batch_tokens = torch.tensor(
                [tokens[i:i+self.ctx_len]], 
                dtype=torch.long, 
                device=self.device
            )

            # Compute the logits and state
            logits_arr, att_shift_states, ffn_shift_states, wkv_states = self.model.forward(
                batch_tokens, att_shift_states, ffn_shift_states, wkv_states
            )

        # Return the logits and state
        return logits_arr[0][-1], { "att_shift_states": att_shift_states, "ffn_shift_states": ffn_shift_states, "wkv_states": wkv_states }
    
    # Forwarding logic, with torch._no_grad() context
    def forward(
            self, tokens:list, 
            stateObj = None
        ):
        with torch.no_grad():
            return self._forward(tokens, stateObj)

    # Sampling logits
    def sample_logits(
            self, logits, 
            prv_tokens=[0], 
            temperature=1.0, top_p=0.9,
            token_ban: list = []
            ):
        # Apply token ban
        for x in token_ban:
            logits[x] = -float("Inf")

        # Handle sampling with temperature
        if temperature > 0.0:
            probs = F.softmax(logits, dim=-1)
            sorted_probs = torch.sort(probs, descending=True)[0]
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1).cpu().numpy()
            cutoff = float(sorted_probs[np.argmax(cumulative_probs > top_p)])
            probs[probs < cutoff] = 0
            if temperature != 1.0:
                probs = probs.pow(1.0 / temperature)
            out = torch.multinomial(probs, num_samples=1)[0]
            return out
        else: 
            # Since the tokenizer sample does not support temp==0
            # we handle this case ourself, by fining the top token
            return torch.argmax(logits, dim=-1).item()

    # Completion API
    def completion(self, 
            prompt, 
            max_tokens: int = 32,
            temperature: float = 1.0,
            top_p: float = 0.9,
            token_ban: list = [],
            start_state = None,
            stream_to_stdout: bool = False,
        ):
        # Encode the context, if its a string
        if isinstance(prompt, str):
            enc = self.encode(prompt)
        # Check if the prompt is a list of tokens
        elif isinstance(prompt, list):
            enc = prompt
        else:
            raise ValueError("Prompt must be a string or a list of tokens")

        # Keep track of the logits and state
        logits = None
        stateObj = start_state

        # For each token, process the state
        logits, stateObj = self.forward(enc, stateObj)

        # # Garbage collect
        # gc.collect()
        # torch.cuda.empty_cache()

        # Generate each token
        full_tokens = enc.copy()
        out_tokens = []
        for i in range(max_tokens):
            ttt = self.sample_logits(
                logits, prv_tokens=full_tokens,
                temperature=temperature, top_p=top_p,
                token_ban=token_ban
            )
            
            # Append the token
            out_tokens.append(ttt)
            full_tokens.append(ttt)
            if stream_to_stdout:
                print(self.decode([ttt]), end="", flush=True)

            # Perform the forward pass
            logits, stateObj = self.forward([ttt], stateObj)

        # Decode the tokens
        out_str = self.decode(out_tokens)

        # # Garbage collect
        # gc.collect()
        # torch.cuda.empty_cache()

        # Return the output string, and state
        return out_str, stateObj
