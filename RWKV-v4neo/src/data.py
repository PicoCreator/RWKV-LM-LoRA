from lightning import LightningDataModule

from torch.utils.data import Dataset

from datasets import load_from_disk, load_dataset
from transformers import PreTrainedTokenizerFast
from multiprocessing import cpu_count

def get_data_module(
        # load_from_disk(dataset_path) param
        data_path: str,
        # load_dataset(path) param
        source: str = None,
        # load_dataset(data_dir) param
        source_data_dir: str = None,
        # Test split of source data, if it was not already done
        test_split: float = 0.1,
        test_split_shuffle: bool = False,
        # Custom tokenizer settings
        tokenizer: str = "neox",
        # Text rechunking size
        text_rechunk_size: int = 2048,
        # ---
        # HF dataset conversion helpers
        # ---
        # Custom 'text' column to support, mostly used for dataset where the 
        # desired train data is in another column (eg. 'code')
        custom_text_key: str = None,
        # Multi column merging support, used for instruct/input/output datasets
        # or similar varients where the input and output are in different columns
        # and need to be merged
        multi_column_keys: list = None,
        multi_column_prefix: list = None,
        multi_column_masking: list = None,
        multi_column_separator: str = None,
        # prompt/completion format masking support
        disable_prompt_mask: bool = False
    ) -> LightningDataModule:
    # Number of max cpu cores
    num_cpus = cpu_count()

    # Source data processing
    if source is not None:
        if tokenizer is None:
            raise ValueError('Tokenizer must be specified if source is specified')

        # Setup the basic load_dataset params
        load_dataset_params = {
            'path': source,
            'num_proc': num_cpus
        }

        # Handle advance params (if set)
        if source_data_dir is not None:
            load_dataset_params['data_dir'] = source_data_dir

        # Load the dataset
        src_dataset = load_dataset(**load_dataset_params)

        # Load the tokenizer
        # according to either its predefined name or its path
        # (defaults to neox)
        if tokenizer == "neox":
            tokenizer_file = "./20B_tokenizer.json"
            tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_file)
        elif tokenizer == "world":
            raise NotImplementedError("World tokenizer not implemented yet")
        else:
            tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer)

        # Multi column merging default values setup
        if multi_column_keys is None:
            multi_column_keys = ['instruction', 'input', 'output']
            multi_column_prefix = ['Instruction:\n', 'Input:\n', 'Output:\n']
            multi_column_masking = [True, True, False]
            multi_column_separator = '\n\n'
        
        # Tokenized encodings for multi column keys
        multi_column_enabled = len(multi_column_keys) > 0
        multi_column_prefix_encodings = []
        multi_column_seperator_encodings = None

        # Process the multi column settings
        if multi_column_enabled:
            # Check if the multi column keys lengths are valid (only if it is enabled)
            if len(multi_column_keys) != len(multi_column_prefix) or len(multi_column_keys) != len(multi_column_masking):
                raise ValueError('Multi column keys, prefix and masking must be the same length')
            # Tokenize the multi column strings
            for i in range(len(multi_column_keys)):
                multi_column_prefix_encodings.append(tokenizer(multi_column_prefix[i]))
            # Tokenize the multi column seperator
            if multi_column_separator is not None and len(multi_column_separator) > 0:
                multi_column_seperator_encodings = tokenizer(multi_column_separator)

        # Maps the dataset record to the tokenized result
        # handles a wide variety of format according to the data configuration
        #
        # - custom text keys
        # - multiple key columns merged
        # - prompt/completion format
        # - text column itself
        #
        # Throws an error, if it failed to process the record
        #
        # This is called for each row record in the dataset
        def map_tokenizer(x):
            # Custom text column support
            if custom_text_key is not None:
                if custom_text_key in x:
                    return tokenizer(x[custom_text_key])
                
            # Multi column merging support
            if multi_column_enabled:
                # Lets count the number of columns we have
                # that have data in them
                num_columns = 0
                for i in range(len(multi_column_keys)):
                    if multi_column_keys[i] in x and x[multi_column_keys[i]] is not None and len(x[multi_column_keys[i]]) > 0:
                        num_columns += 1
                # If we have more than 1 column, we will have to merge them
                if num_columns > 1:
                    # Array of output values we will return
                    input_ids = []
                    token_type_ids = []
                    attention_mask = []

                    # First item flag
                    is_first_item = True

                    # Lets loop through each column
                    for i in range(len(multi_column_keys)):
                        # And process the column if it has data
                        if multi_column_keys[i] in x and x[multi_column_keys[i]] is not None and len(x[multi_column_keys[i]]) > 0:
                            # Add the seperator if this is not the first item
                            if not is_first_item and multi_column_seperator_encodings is not None:
                                input_ids += multi_column_seperator_encodings['input_ids']
                                token_type_ids += multi_column_seperator_encodings['token_type_ids']
                                attention_mask += multi_column_seperator_encodings['attention_mask']
                            
                            # Add the prefix
                            input_ids += multi_column_prefix_encodings[i]['input_ids']
                            token_type_ids += multi_column_prefix_encodings[i]['token_type_ids']
                            attention_mask += multi_column_prefix_encodings[i]['attention_mask']

                            # Tokenize the column
                            column_encodings = tokenizer(x[multi_column_keys[i]])

                            # Add the column
                            input_ids += column_encodings['input_ids']
                            token_type_ids += column_encodings['token_type_ids']

                            # Add the attention mask if masking is enabled
                            if multi_column_masking[i]:
                                attention_mask += ([1] * len(column_encodings['input_ids']))
                            else:
                                attention_mask += column_encodings['attention_mask']
                    
                    # Return the merged columns
                    return {
                        'input_ids': input_ids,
                        'token_type_ids': token_type_ids,
                        'attention_mask': attention_mask
                    }

            # Prompt completion support
            if 'prompt' in x and 'completion' in x:
                # Array of output values we will return
                input_ids = None
                token_type_ids = None
                attention_mask = None

                # Tokenize both prompt and completion
                # Note that the tokenizer will process and return the input_ids in batches
                prompt_encodings = tokenizer(x['prompt'])
                completion_encodings = tokenizer(x['completion'])

                # Join the two input_ids lists
                input_ids = prompt_encodings['input_ids'] + completion_encodings['input_ids']
                # Join the two token_type_ids lists
                token_type_ids = prompt_encodings['token_type_ids'] + completion_encodings['token_type_ids']
                # Setup the attention mask, 0 for prompt, 1 for completion, if masking is enabled
                if disable_prompt_mask:
                    attention_mask = ([1] * len(prompt_encodings['input_ids']) + [1] * len(completion_encodings['input_ids']))
                else:
                    attention_mask = ([0] * len(prompt_encodings['input_ids']) + [1] * len(completion_encodings['input_ids']))

                # Prepare and return the output object
                ret = {
                    'input_ids': input_ids,
                    'token_type_ids': token_type_ids,
                    'attention_mask': attention_mask,
                }
                return ret
            
            # Fallback to standard text tokenization
            if 'text' in x:
                return tokenizer(x['text'])
            
            raise ValueError('Invalid dataset format, must contain either the configured "multi column" or prompt/completion or text')

        # Map the dataset to the tokenizer, removing the old text column
        src_dataset = src_dataset.map(map_tokenizer, batched=False, num_proc=num_cpus)

        # See if rechunking is needed, this is useful only for raw "text" based datasets
        # where we would need to split them into "digestable" context length sizes
        # (this function will break otherwise, due to change in the sample sizes)
        if source == "text" and text_rechunk_size > 0:
            # Get the newline token
            newline_tokenSet = tokenizer(["\n"])

            # The rechunking function
            def rechunk_text(x):
                # Full Raw values that we will need to "rechunk"
                full_input_ids = []
                full_token_type_ids = []
                full_attention_mask = []

                # Loop through the x input, and build the raw values
                for i in range(len(x["input_ids"])):
                    # Get the respective values and push them to the 
                    # raw value array, effectively merging the arrays together
                    # with the newline token in between
                    full_input_ids += x["input_ids"][i] + newline_tokenSet["input_ids"][0]
                    full_token_type_ids += x["token_type_ids"][i] + newline_tokenSet["token_type_ids"][0]
                    full_attention_mask += x["attention_mask"][i] + newline_tokenSet["attention_mask"][0]
                
                # Total length, and sample count
                # note that thte "remainder" will be discarded
                total_len = len(full_input_ids)
                total_samples = total_len // text_rechunk_size

                # The output arrays
                out_input_ids = []
                out_token_type_ids = []
                out_attention_mask = []

                # Generate the output arrays
                for i in range(total_samples):
                    # Calculate the start and end of the sample
                    start = i * text_rechunk_size
                    end = start + text_rechunk_size

                    # Push the sample to the output arrays
                    out_input_ids.append(full_input_ids[start:end])
                    out_token_type_ids.append(full_token_type_ids[start:end])
                    out_attention_mask.append(full_attention_mask[start:end])
                
                # Prepare and return the output object
                ret = {
                    'input_ids': out_input_ids,
                    'token_type_ids': out_token_type_ids,
                    'attention_mask': out_attention_mask,
                }
                return ret

            # Perform the rechunking
            src_dataset = src_dataset.map(rechunk_text, batched=True, 
                                          batch_size=text_rechunk_size*10,
                                          remove_columns=['text'],
                                          num_proc=num_cpus)

        # Check if the dataset does not have a test split
        # and if so, perform the split
        if 'test' not in src_dataset.keys():
            src_dataset = src_dataset['train'].train_test_split(test_size=test_split,shuffle=test_split_shuffle)
        
        # Save the dataset to disk
        src_dataset.save_to_disk(data_path)

    # Load the dataset as per normal 
    dataset = load_from_disk(data_path).with_format('torch')
    return LightningDataModule.from_datasets(dataset['train'], dataset['test'], num_workers=num_cpus)
