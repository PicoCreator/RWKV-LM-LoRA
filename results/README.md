# RWKV memory experiment

**!Important disclaimer note:** 

This test the extremes of RWKV memory with random words, where there is NO pattern, or association with existing learned concepts. In general RWKV is able to memorize/learn concepts or even code that uses more token then this experiment has done. 

For example, users have shown RWKV to rewrite letters / answer Q&A with reference materials / etc with larger context sizes then what is provided here

You can think of this as at best, how much "compressed" knowledge can RWKV memorize. Or alternatively how much RWKV can store in the extream worse case scenerio

For example you as a human have a much easier time roughly memorizing a 500 word story, in approximate, then 50 words randomly chosen with no pattern.

Finally this is using a new finetune process which may have bugs, and the testing/training methodology may have flaws that needs to be refined.

**Experiment note:**

For some reason reason the finetuned model goes crazy once it encounters the " on" token, i have blocked this token, and removed it from the test data.

The eval script can be found at `eval_model_memory_guided.py`, for more details of the whole memory test process you can see the `experiment-process.md`

The guided eval script, uses the memory task prompt that raven was trained with. It also "auto-correct" the evaluation process with the correct token, and measure where it failed.
This helps trigger, the weaker associative memories, even if the model accidentally fall off track early (instead of being permenantly off the rails), while this means the matched token score might be slightly higher then actual day-to-day use, it is also a more accurate measurement of the model overall memory capacity.

## RWKV 3B memory results

**Native raven 3B results**
- starts to lose accuracy after 10 tokens (still 90%+ accurate)
- sharp drop after 40 tokens

**Finetuned telephone 3B**
- perfect memory until about 35 tokens
- minor memory loss ('s) until 55 tokens
- sharp drop after 55 tokens

## RWKV 7B memory results

**Native raven 7B**
- (skip) wierd behaviour under 20 tokens
- perfect memory from 25-35 tokens
- minor 1 to 3 word drop from 40-50 tokens
- major drop after 50 tokens

## RWKV 14B memory results

**Native raven 7B**
- perfect memory until 50 tokens
- drop off after 50 tokens
