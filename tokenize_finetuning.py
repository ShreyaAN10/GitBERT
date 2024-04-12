import csv
import time
import torch
import pickle
from tqdm import tqdm
from transformers import BertTokenizer
import sentencepiece as spm
import argparse

import logging
logging.disable(logging.WARNING)

def read_tsv(tsv_path):
    """
    Read data from a TSV file where each line contains inputs and targets separated by a tab.

    Args:
        tsv_path (str): Path to the TSV file.

    Returns:
        sentences (list): List of input sentences.
    """
    sentences = []
    with open(tsv_path, 'r') as tsv_file:
        for line in tsv_file:
            input_text, target_text = line.strip().split('\t')
            input_text = input_text + 'NEXT' + target_text
            sentences.append(input_text[-512:])
            
    return sentences

parser = argparse.ArgumentParser(description="BERT model fine-tuning script")
parser.add_argument("mode", choices=["train", "test", "eval"], help="Mode for script execution")
parser.add_argument("task", choices=["ns", "jc"], help="ns for Next-Sentence, jc for Job-Completion")
args = parser.parse_args()

if args.mode == 'train':
    if args.task == 'ns':
        sentences = read_tsv('./ghwcom-dataset/Fine-tuning/Next-Sentence/TSV/abstracted/train.tsv')
        output_file_path = "./ghwcom-dataset/Fine-tuning/Next-Sentence/tokenized/next_sentence_tokenized_train.pkl"
    elif args.task == 'jc':
        sentences = read_tsv('./ghwcom-dataset/Fine-tuning/Job-Completion/TSV/abstracted/train.tsv')
        output_file_path = "./ghwcom-dataset/Fine-tuning/Job-Completion/tokenized/jc_tokenized_train.pkl"
elif args.mode == 'test':
    if args.task == 'ns':
        sentences = read_tsv('./ghwcom-dataset/Fine-tuning/Next-Sentence/TSV/abstracted/test.tsv')
        output_file_path = "./ghwcom-dataset/Fine-tuning/Next-Sentence/tokenized/next_sentence_tokenized_test.pkl"
    elif args.task == 'jc':
        sentences = read_tsv('./ghwcom-dataset/Fine-tuning/Job-Completion/TSV/abstracted/test.tsv')
        output_file_path = "./ghwcom-dataset/Fine-tuning/Job-Completion/tokenized/jc_tokenized_test.pkl"
elif args.mode == 'eval':
    if args.task == 'ns':
        sentences = read_tsv('./ghwcom-dataset/Fine-tuning/Next-Sentence/TSV/abstracted/eval.tsv')
        output_file_path = "./ghwcom-dataset/Fine-tuning/Next-Sentence/tokenized/next_sentence_tokenized_eval.pkl"
    elif args.task == 'jc':
        sentences = read_tsv('./ghwcom-dataset/Fine-tuning/Job-Completion/TSV/abstracted/eval.tsv')
        output_file_path = "./ghwcom-dataset/Fine-tuning/Job-Completion/tokenized/jc_tokenized_eval.pkl"

# Instantiate tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Get time before tokenization
start_time = time.time()
print(f'Tokenizing...')
inputs = tokenizer(sentences, return_tensors='pt',
                    max_length=512, truncation=True, padding='max_length')
# Get time after tokenization
end_time = time.time()
# Calculate the time taken for tokenization
time_taken = end_time - start_time
print("Time taken for tokenization:", time_taken, "seconds")

# Create labels
inputs['labels'] = inputs.input_ids.detach().clone()
print(f'inputs1: {inputs.keys()}')

print(f'Masking...')
masks = []
error = 0
for i in tqdm(range(len(sentences))):
    indices = []
    for idx, val in enumerate(inputs['input_ids'][i]):
        if val == 2279:
            # this is the NEXT token
            indices.append(idx)
        if val == 102:
            # this is the [SEP] token
            indices.append(idx)
    if len(indices) < 2:
        error += 1
        continue
    num_of_masks = indices[1] - indices[0] - 1

    start_idx = indices[0] + 1
    while start_idx < indices[1]:
        # mask all tokens from NEXT to [SEP]
        inputs['input_ids'][i][start_idx] = 103
        start_idx += 1

print(f'Number of disregarded instances: {error}')
# Save inputs to a file
print(f'Saving inputs to {output_file_path}...')
# Save the dictionary to a pickle file
with open(output_file_path, "wb") as pickle_file:
    pickle.dump(inputs, pickle_file)

