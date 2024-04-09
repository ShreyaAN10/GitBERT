import json
import pickle
import time
import torch
import random
from transformers import BertTokenizer

FOR_MLM = True

def load_and_merge_json(json1_path, json2_path, json_path):
    """
    Load contents of two JSON files, merge them, and write the merged content back to a new JSON file.

    Args:
        json1_path (str): Path to the first JSON file.
        json2_path (str): Path to the second JSON file.
        json_path (str): Path to write the merged JSON content.
    """
    # Load contents of both JSON files
    with open(json1_path, "r") as file:
        json1_content = json.load(file)

    with open(json2_path, "r") as file:
        json2_content = json.load(file)

    # Append json2_content to json1_content
    json_content = json1_content + json2_content
    print(len(json1_content), len(json2_content), len(json_content))

    # Write the merged content back to json1
    with open(json_path, "w") as file:
        json.dump(json_content, file, indent=4)  # indent for pretty formatting (optional)

def process_json_data(json_path):
    """
    Process merged JSON data.

    Args:
        json_path (str): Path to the merged JSON file.
    """
    # Load the JSON data
    with open(json_path, 'r') as json_file:
        data = json.load(json_file)

    # Extract "jsonized-data" from each entry and store in a dictionary with "input" key
    inputs_bag = [entry['jsonized-data'] for entry in data]

    # Print the first entry of the list
    print(inputs_bag[0])

    bag_size = len(inputs_bag)

    sentences_a = []
    sentences_b = []
    labels = []

    count = 0
    for script in inputs_bag:
        count += 1
        sentences = script.split(', ')
        num_sentences = len(sentences)
        if num_sentences > 1:
            start = random.randint(0, num_sentences-2)
            sentences_a.append(sentences[start])
            if random.random() > 0.5:
                sentences_b.append(sentences[start+1])
                labels.append(0)
            else:
                sentences_b.append(inputs_bag[random.randint(0, bag_size-1)])
                labels.append(1)

    return sentences_a, sentences_b, labels

# Paths to your JSON files
json1_path = "./ghwcom-dataset/Pre-training/yaml.json"
json2_path = "./ghwcom-dataset/Pre-training/actions.json"
json_path = "./ghwcom-dataset/Pre-training/actions_yaml.json"

# Call load_and_merge_json function
load_and_merge_json(json1_path, json2_path, json_path)

# Call process_json_data function
sentences_a, sentences_b, labels = process_json_data(json_path)

sentences = sentences_a + sentences_b

# Instantiate tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Get time before tokenization
start_time = time.time()
print(f'Tokenizing...')

if not FOR_MLM:
    # Use this for NEXT SENTENCE training
    inputs = tokenizer(sentences_a, sentences_b, return_tensors='pt',
                        max_length=512, truncation=True, padding='max_length')
elif FOR_MLM:
    # Use this for NEXT SENTENCE training
    inputs = tokenizer(sentences, return_tensors='pt',
                        max_length=512, truncation=True, padding='max_length')
# Get time after tokenization
end_time = time.time()
# Calculate the time taken for tokenization
time_taken = end_time - start_time
print("Time taken for tokenization:", time_taken, "seconds")

print(f'1: {inputs.keys()}')

if not FOR_MLM:
    inputs['next_sentence_label'] = torch.LongTensor([labels]).T

print(torch.LongTensor([labels]).T)
print(torch.LongTensor([labels]))
print(torch.LongTensor(labels))
print(f'2: {inputs.keys()}')

inputs['labels'] = inputs.input_ids.detach().clone()
print(f'3: {inputs.keys()}')

# Save inputs to a file
if not FOR_MLM:
    output_file_path = "./ghwcom-dataset/Pre-training/tokenized_NS_actions_yaml.pkl"
elif FOR_MLM:
    output_file_path = "./ghwcom-dataset/Pre-training/tokenized_MLM_actions_yaml.pkl"
print(f'Saving inputs to {output_file_path}...')
# Save the dictionary to a JSON file
with open(output_file_path, "wb") as pickle_file:
    pickle.dump(inputs, pickle_file)