from lightning import LightningDataModule

from torch.utils.data import Dataset

from datasets import load_from_disk, load_dataset
from transformers import PreTrainedTokenizerFast


def get_data_module(data_path: str,
                    source: str = None,
                    tokenizer: str = None) -> LightningDataModule:
    if source is not None:

        if tokenizer is None:
            raise ValueError('Tokenizer must be specified if source is specified')

        src_dataset = load_dataset(source, split='train')
        tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer)

        def map_tokenizer(x):
            if 'prompt' in x and 'completion' in x:
                # Array of output valeus we will return
                input_ids = []
                token_type_ids = []
                attention_mask = []

                # Tokenize both prompt and completion
                # Note that the tokenizer will process and return the input_ids in batches
                prompt_encodings = tokenizer(x['prompt'])
                completion_encodings = tokenizer(x['completion'])

                # # (debugging) Print the prompt and completion encoding objects 
                # print("prompt: ", x['prompt'])
                # print("prompt: ", x['completion'])
                # print("prompt_encodings: ", prompt_encodings)
                # print("completion_encodings: ", completion_encodings)

                # Important note, prompt_encodings['input_ids'] are list, containing list of the actual values
                # so we need to process them accordingly (batch processing)
                for i in range(len(prompt_encodings['input_ids'])):
                    # Join the two input_ids lists
                    input_ids.append(prompt_encodings['input_ids'][i] + completion_encodings['input_ids'][i])

                    # Join the two token_type_ids lists
                    token_type_ids.append(prompt_encodings['token_type_ids'][i] + completion_encodings['token_type_ids'][i])

                    # Setup the attention mask, 0 for prompt, 1 for completion
                    attention_mask.append([0] * len(prompt_encodings['input_ids'][i]) + [1] * len(completion_encodings['input_ids'][i]))

                # Prepare the output object
                ret = {
                    'input_ids': input_ids,
                    'token_type_ids': token_type_ids,
                    'attention_mask': attention_mask,
                }
                
                # # (debugging) Print the output object
                # print("ret: ", ret)

                return ret
            else:
                # Fallback to standard text tokenization
                return tokenizer(x['text'])

        src_dataset = src_dataset.map(map_tokenizer, batched=True)
        src_dataset = src_dataset.train_test_split(test_size=0.1,
                                                   shuffle=False)
        src_dataset.save_to_disk(data_path)

    dataset = load_from_disk(data_path).with_format('torch')
    return LightningDataModule.from_datasets(dataset['train'], dataset['test'])
