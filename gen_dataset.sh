#!/bin/bash

# This script is used to generate the dataset for the project

# From 100 to 1000 words
node ./gen_dataset_file.js ./dataset/word-100-count-100.jsonl 100 100 &
node ./gen_dataset_file.js ./dataset/word-200-count-100.jsonl 200 100 &
node ./gen_dataset_file.js ./dataset/word-300-count-100.jsonl 300 100 &
node ./gen_dataset_file.js ./dataset/word-400-count-100.jsonl 400 100 &
node ./gen_dataset_file.js ./dataset/word-500-count-100.jsonl 500 100 &
node ./gen_dataset_file.js ./dataset/word-600-count-100.jsonl 600 100 &
node ./gen_dataset_file.js ./dataset/word-700-count-100.jsonl 700 100 &
node ./gen_dataset_file.js ./dataset/word-800-count-100.jsonl 800 100 &
node ./gen_dataset_file.js ./dataset/word-900-count-100.jsonl 900 100 &
node ./gen_dataset_file.js ./dataset/word-1000-count-100.jsonl 1000 100 &

# From 1000 to 10000 words
node ./gen_dataset_file.js ./dataset/word-1k-count-1k.jsonl 1000 1000 &
node ./gen_dataset_file.js ./dataset/word-2k-count-1k.jsonl 2000 1000 &
node ./gen_dataset_file.js ./dataset/word-3k-count-1k.jsonl 3000 1000 &
node ./gen_dataset_file.js ./dataset/word-4k-count-1k.jsonl 4000 1000 &
node ./gen_dataset_file.js ./dataset/word-5k-count-1k.jsonl 5000 1000 &
node ./gen_dataset_file.js ./dataset/word-6k-count-1k.jsonl 6000 1000 &
node ./gen_dataset_file.js ./dataset/word-7k-count-1k.jsonl 7000 1000 &
node ./gen_dataset_file.js ./dataset/word-8k-count-1k.jsonl 8000 1000 &
node ./gen_dataset_file.js ./dataset/word-9k-count-1k.jsonl 9000 1000 &
node ./gen_dataset_file.js ./dataset/word-10k-count-1k.jsonl 10000 1000 &

wait
echo "## Done ##"
