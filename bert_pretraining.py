import random
import time
import json
import torch
import pickle
from transformers import AdamW
from tqdm import tqdm
from transformers import BertForPreTraining, BertForMaskedLM

N_EPOCHS = 1

class PreTrainingDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings
    def __getitem__(self, idx):
        return {key: torch.Tensor(val[idx]) for key, val in self.encodings.items()}
    def __len__(self):
        return len(self.encodings.input_ids)

# Path to the JSON file
input_file_path = "./ghwcom-dataset/Pre-training/tokenized_MLM_actions_yaml.pkl"

# Read data from the pickle file into a new dictionary
with open(input_file_path, "rb") as pickle_file:
    inputs = pickle.load(pickle_file)

# Instantiate BERT model
model = BertForMaskedLM.from_pretrained('bert-base-uncased')

# Mask 15% of the input_ids
rand = torch.rand(inputs.input_ids.shape)
print(rand < 0.15)
# Do not mask special tokens ([CLS], [SEP], [PAD])
mask_arr = (rand < 0.15) * (inputs.input_ids != 101) * (inputs.input_ids != 102) * (inputs.input_ids != 0)
print(f'mask_arr {mask_arr}')
# Extract indices of MASK in mask_arr to mask input_ids in inputs
for i in range(mask_arr.shape[0]):
    selection = torch.flatten(mask_arr[i].nonzero()).tolist()
    # print(f'selection {selection}')
    inputs.input_ids[i, selection] = 103

# Transform data into PyTorch dataset
dataset = PreTrainingDataset(inputs)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

device = "cuda" if torch.cuda.is_available else "cpu"
print(f'Using {device}...')
model.to(device)

results = {
    "epoch": [],
    "loss": []
}

optim = AdamW(model.parameters(), lr=5e-5)
for epoch in range(N_EPOCHS):
    model.train()
    running_loss = 0.0 # Initialize running loss
    loop = tqdm(dataloader, leave=True)
    for batch in loop:
        optim.zero_grad()
        input_ids = batch['input_ids'].to(device)
        token_type_ids = batch['token_type_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        output = model(input_ids, 
                        token_type_ids=token_type_ids, 
                        attention_mask=attention_mask,
                        labels=labels)
        loss = output.loss
        loss.backward()
        optim.step()
        # Accumulate running loss
        running_loss += loss.item()
        loop.set_description(f'Epoch {epoch}')
        loop.set_postfix(loss=loss.item())
    epoch_loss = running_loss / len(dataloader)
    results["epoch"].append(epoch)
    results["loss"].append(epoch_loss)
    print(f'Epoch {epoch} Loss {epoch_loss}')

# Define paths for saving the model and tokenizer
sd_path = "./models/pretrained_MLM_bert_10e_state_dict.pth"
model_path = "./models/pretrained_MLM_bert_10e_model.pth"
print(f'Saving model to {model_path}...')
# Save the model's state_dict
torch.save(model.state_dict(), sd_path)
# Save the model
torch.save(model, model_path)

# Save results to a file
output_file_path = "./models/pretrained_MLM_bert_10e_results.pkl"
print(f'Saving results to {output_file_path}...')
# Save the dictionary to a pickle file
with open(output_file_path, "wb") as pickle_file:
    pickle.dump(results, pickle_file)







