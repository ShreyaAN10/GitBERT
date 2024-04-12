import torch
import pickle
import argparse
import logging
import csv
from torch.utils.data import Dataset, DataLoader
from transformers import AdamW, BertForMaskedLM, BertForPreTraining, BertConfig
from tqdm import tqdm
from transformers.optimization import get_scheduler

from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

logging.disable(logging.WARNING)

N_EPOCHS = 10

def read_tsv(file_path):
    with open(file_path, 'r', encoding='utf-8') as tsvfile:
        reader = csv.reader(tsvfile, delimiter='\t')
        data = [(row[0], row[1]) for row in reader]
    return data

class FineTuningDataset(Dataset):
    def __init__(self, encodings):
        self.encodings = encodings
    def __getitem__(self, idx):
        return {key: torch.Tensor(val[idx]) for key, val in self.encodings.items()}
    def __len__(self):
        return len(self.encodings.input_ids)

def train(args, model, train_dataloader, optimizer, scheduler, device):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    iter = 0
    this_results = {
        "batch": [],
        "loss": [],
        "accuracy": []
    }
    batch_iter = 0
    for batch in tqdm(train_dataloader, desc="Training", leave=False):
        optimizer.zero_grad()
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
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()
        logits = output.logits
        # Apply a mask to retain only the logits corresponding to the masked tokens
        masked_positions = (input_ids == tokenizer.mask_token_id)
        # Get the logits for the masked positions
        masked_logits = logits[masked_positions]
        # Get the predicted labels for the masked tokens
        predicted_labels = torch.argmax(masked_logits, dim=-1)
        # Get the ground truth labels for the masked tokens
        ground_truth_labels = labels[masked_positions]
        # Calculate the accuracy for the masked tokens
        masked_accuracy = torch.sum(predicted_labels == ground_truth_labels).item() / masked_positions.sum().item()
        total_correct += torch.sum(predicted_labels == ground_truth_labels).item()
        total_samples += masked_positions.sum().item()
        print(f'Evaluating batch: Correct Predictions: {torch.sum(predicted_labels == ground_truth_labels).item()} | Total Samples: {masked_positions.sum().item()} | Batch Accuracy: {masked_accuracy}')
        this_results["batch"].append(iter)
        this_results["loss"].append(loss.item())
        this_results["accuracy"].append(masked_accuracy)
    print(f'[*] Training')
    batch_iter += 1
    if batch_iter == 1:
        path = f"./models/JC_BASE_{args.scheduler}_intermediate_train.pkl"
        with open(path, "wb") as pickle_file:
            pickle.dump(this_results, pickle_file)
    print(this_results)
    accuracy = total_correct / total_samples
    return total_loss / len(train_dataloader), accuracy

def evaluate(args, model, dataloader, device):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    iter = 0
    this_results = {
        "batch": [],
        "loss": [],
        "accuracy": []
    }
    batch_iter = 0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            input_ids = batch['input_ids'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids, 
                        token_type_ids=token_type_ids, 
                        attention_mask=attention_mask, 
                        labels=labels)
            loss = outputs.loss
            total_loss += loss.item()
            logits = outputs.logits
            # Apply a mask to retain only the logits corresponding to the masked tokens
            masked_positions = (input_ids == tokenizer.mask_token_id)
            # Get the logits for the masked positions
            masked_logits = logits[masked_positions]
            # Get the predicted labels for the masked tokens
            predicted_labels = torch.argmax(masked_logits, dim=-1)
            # Get the ground truth labels for the masked tokens
            ground_truth_labels = labels[masked_positions]
            # Calculate the accuracy for the masked tokens
            masked_accuracy = torch.sum(predicted_labels == ground_truth_labels).item() / masked_positions.sum().item()
            total_correct += torch.sum(predicted_labels == ground_truth_labels).item()
            total_samples += masked_positions.sum().item()
            print(f'Evaluating batch: Correct Predictions: {torch.sum(predicted_labels == ground_truth_labels).item()} | Total Samples: {masked_positions.sum().item()} | Batch Accuracy: {masked_accuracy}')
            this_results["batch"].append(iter)
            this_results["loss"].append(loss.item())
            this_results["accuracy"].append(masked_accuracy)
    print(f'[*] Evaluation')
    batch_iter += 1
    if batch_iter == 1:
        path = f"./models/JC_BASE_{args.scheduler}_intermediate_eval.pkl"
        with open(path, "wb") as pickle_file:
            pickle.dump(this_results, pickle_file)
    print(this_results)
    accuracy = total_correct / total_samples
    return total_loss / len(dataloader), accuracy

def main(args):
    
    if args.model == 'base':
        print(f'Loading BERT base model...')
        model = BertForMaskedLM.from_pretrained('bert-base-uncased')
    elif args.model == 'pretrained':
        print(f'Loading pretrained BERT model...')
        model = torch.load("/home/s4an/bert-autocom/models/pretrained_MLM_bert_10e_model.pth")

    if args.task == 'ns':
        print(f'Generating inputs for Next-Sentence task...')
        train_inputs_file_path = './ghwcom-dataset/Fine-tuning/Next-Sentence/tokenized/next_sentence_tokenized_train.pkl'
        with open(train_inputs_file_path, "rb") as pickle_file:
            train_inputs = pickle.load(pickle_file)
        train_dataset = FineTuningDataset(train_inputs)
        train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)

        test_inputs_file_path = './ghwcom-dataset/Fine-tuning/Next-Sentence/tokenized/next_sentence_tokenized_test.pkl'
        with open(test_inputs_file_path, "rb") as pickle_file:
            test_inputs = pickle.load(pickle_file)
        test_dataset = FineTuningDataset(test_inputs)
        test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=True)

        eval_inputs_file_path = './ghwcom-dataset/Fine-tuning/Next-Sentence/tokenized/next_sentence_tokenized_eval.pkl'
        with open(eval_inputs_file_path, "rb") as pickle_file:
            eval_inputs = pickle.load(pickle_file)
        eval_dataset = FineTuningDataset(eval_inputs)
        eval_dataloader = DataLoader(eval_dataset, batch_size=32, shuffle=True)
    elif args.task == 'jc':
        print(f'Generating inputs for Job-Completion task...')
        train_inputs_file_path = './ghwcom-dataset/Fine-tuning/Job-Completion/tokenized/jc_tokenized_train.pkl'
        with open(train_inputs_file_path, "rb") as pickle_file:
            train_inputs = pickle.load(pickle_file)
        train_dataset = FineTuningDataset(train_inputs)
        train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)

        test_inputs_file_path = './ghwcom-dataset/Fine-tuning/Job-Completion/tokenized/jc_tokenized_test.pkl'
        with open(test_inputs_file_path, "rb") as pickle_file:
            test_inputs = pickle.load(pickle_file)
        test_dataset = FineTuningDataset(test_inputs)
        test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=True)

        eval_inputs_file_path = './ghwcom-dataset/Fine-tuning/Job-Completion/tokenized/jc_tokenized_eval.pkl'
        with open(eval_inputs_file_path, "rb") as pickle_file:
            eval_inputs = pickle.load(pickle_file)
        eval_dataset = FineTuningDataset(eval_inputs)
        eval_dataloader = DataLoader(eval_dataset, batch_size=32, shuffle=True)

    optimizer = AdamW(model.parameters(), lr=5e-5)
    device = "cuda" if torch.cuda.is_available else "cpu"
    print(f'Using {device}...')
    model.to(device)

    if args.scheduler == 'constant':
        print(f'Training with constant learning rate scheduler...')
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=1.0)
    elif args.scheduler == 'polynomial':
        print(f'Training with polynomial learning rate scheduler...')
        decay_steps = 10000
        end_learning_rate = 0.001
        scheduler = torch.optim.lr_scheduler.PolynomialLR(optimizer, decay_steps, end_learning_rate)
    elif args.scheduler == 'inverse_sqrt':
        print(f'Training with inverse square root learning rate scheduler...')
        warmup_steps = 1000
        scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=warmup_steps, num_training_steps=10000)
    elif args.scheduler == 'slanted_triangular':
        print(f'Training with slanted triangular learning rate scheduler...')
        num_warmup_steps = 1000
        num_training_steps = 10000
        scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)

    results = {
            "epoch": [],
            "train_loss": [],
            "train_accuracy": [],
            "val_loss": [],
            "val_accuracy": [],
            "test_loss": [],
            "test_accuracy": []
        }
    for epoch in range(N_EPOCHS):
        train_loss, train_accuracy = train(args, model, train_dataloader, optimizer, scheduler, device)
        val_loss, val_accuracy = evaluate(args, model, eval_dataloader, device)
        
        print(f"Epoch {epoch + 1}: Train Loss: {train_loss}, Train Accuracy: {train_accuracy}, Val Loss: {val_loss}, Val Accuracy: {val_accuracy}")
        
        results["epoch"].append(epoch+1)
        results["train_loss"].append(train_loss)
        results["train_accuracy"].append(train_accuracy)
        results["val_loss"].append(val_loss)
        results["val_accuracy"].append(val_accuracy)
    print(results)

    test_loss, test_accuracy = evaluate(args, model, test_dataloader, device)
    print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")
    results["test_loss"].append(test_loss)
    results["test_accuracy"].append(test_accuracy)
    print(results)

    model_path = f"./models/{(args.task).upper()}/{(args.task).upper()}_finetuned_bert_{(args.model).upper()}_{args.scheduler}_10e.pth"
    print(f'Saving model to {model_path}...')
    torch.save(model.state_dict(), model_path)

    output_file_path = f"./models/{(args.task).upper()}/{(args.task).upper()}_finetuned_bert_{(args.model).upper()}_{args.scheduler}_10e.pkl"
    print(f'Saving results to {output_file_path}...')
    with open(output_file_path, "wb") as pickle_file:
        pickle.dump(results, pickle_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BERT fine-tuning with different learning rate schedulers")
    parser.add_argument("-scheduler", type=str, choices=['constant', 'polynomial', 'inverse_sqrt', 'slanted_triangular'], help="Learning rate scheduler")
    parser.add_argument("-task", choices=["ns", "jc"], help="ns for Next-Sentence, jc for Job-Completion")
    parser.add_argument("-model", choices=["base", "pretrained"], help="ns for Next-Sentence, jc for Job-Completion")
    args = parser.parse_args()
    main(args)
