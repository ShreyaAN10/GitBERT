import torch
import argparse
import pickle
from tqdm import tqdm
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu
from nltk.tokenize import word_tokenize
from torch.utils.data import Dataset, DataLoader
from transformers import BertConfig, BertForMaskedLM, BertTokenizer

class FineTuningDataset(Dataset):
    def __init__(self, encodings):
        self.encodings = encodings
    def __getitem__(self, idx):
        return {key: torch.Tensor(val[idx]) for key, val in self.encodings.items()}
    def __len__(self):
        return len(self.encodings.input_ids)

def calculate_rouge_l(predicted_texts, reference_texts):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    scores = scorer.score(predicted_texts, reference_texts)
    return scores

def calculate_bleu_score(predicted_tokens, target_tokens):
    # Tokenize the strings
    predicted_tokens = word_tokenize(predicted_tokens.lower())
    target_tokens = word_tokenize(target_tokens.lower())
    # Calculate BLEU score
    bleu_score = sentence_bleu([target_tokens], predicted_tokens)
    return bleu_score

device = "cuda" if torch.cuda.is_available() else "cpu"

parser = argparse.ArgumentParser(description="Calculate Rogue-L score")
parser.add_argument("-task", choices=["ns", "jc"], help="ns for Next-Sentence, jc for Job-Completion")
parser.add_argument("-model", choices=["base", "pretrained"], help="ns for Next-Sentence, jc for Job-Completion")
args = parser.parse_args()

if args.model == 'base':
    print(f'Loading BERT base model...')
    model = BertForMaskedLM.from_pretrained('bert-base-uncased')
elif args.model == 'pretrained':
    print(f'Loading pretrained BERT model...')
    # Import fine-tuned model
    config = BertConfig.from_pretrained("bert-base-uncased")
    if args.task == 'ns':
        state_dict = "./models/NS/finetuned_bert_PRETRAINED_constant_10e.pth"
    elif args.task == 'jc':
        state_dict = "./models/JC/JC_finetuned_bert_PRETRAINED_constant_10e.pth"
    model = BertForMaskedLM(config)
    # Load the state dictionary from the saved file
    model.load_state_dict(torch.load(state_dict))

model.to(device)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

if args.task == 'ns':
    test_inputs_file_path = './ghwcom-dataset/Fine-tuning/Next-Sentence/tokenized/next_sentence_tokenized_test.pkl'
    with open(test_inputs_file_path, "rb") as pickle_file:
        test_inputs = pickle.load(pickle_file)
    test_dataset = FineTuningDataset(test_inputs)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True)
if args.task == 'jc':
    test_inputs_file_path = './ghwcom-dataset/Fine-tuning/Job-Completion/tokenized/jc_tokenized_test.pkl'
    with open(test_inputs_file_path, "rb") as pickle_file:
        test_inputs = pickle.load(pickle_file)
    test_dataset = FineTuningDataset(test_inputs)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True)

# Test loop
tokens = {
    'preds': [],
    'targs': []
}
model.eval()
from tqdm import tqdm
with torch.no_grad():
    test_loop = tqdm(test_dataloader, leave=True)
    err_count = 0
    for batch in test_loop:
        iter += 1
        input_ids = batch['input_ids'].to(device)
        token_type_ids = batch['token_type_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        outputs = model(input_ids, 
                       token_type_ids=token_type_ids, 
                       attention_mask=attention_mask, 
                       labels=labels)
        # Process the outputs
        logits = outputs.logits
        # Apply softmax to convert logits to probabilities
        probabilities = torch.softmax(outputs.logits, dim=-1)
        # Find the token ID with the highest probability for the masked position
        predicted_token_ids = torch.argmax(probabilities, dim=-1)
        # Create a mask to filter out tokens equal to 101 or 102
        p_mask = (predicted_token_ids != 101)
        preds = predicted_token_ids[p_mask]
        t_mask = (labels != 101)
        targs = labels[t_mask]
        # Find indices of occurrences of "next" token
        p_indices = torch.where(preds == 2279)[0]
        t_indices = torch.where(targs == 2279)[0]
        if len(p_indices) > 0:
            p_next_idx = p_indices[-1]
            t_next_idx = t_indices[-1]
            p_pad_indices = torch.where(preds == 102)[0]
            t_pad_indices = torch.where(targs == 102)[0]
            if len(p_pad_indices) > 0:
                p_pad_idx = p_pad_indices[0]
                t_pad_idx = t_pad_indices[0]
                preds = preds[p_next_idx+1:p_pad_idx]
                targs = targs[t_next_idx+1:t_pad_idx]
                # Decode the predicted token IDs back into words
                predicted_tokens = tokenizer.decode(preds)
                target_tokens = tokenizer.decode(targs)
                tokens["preds"].append(predicted_tokens)
                tokens["targs"].append(target_tokens)
            else:
                err_count += 1
        else:
            err_count += 1

print(f'Errored: {err_count}')

f_measures_L = []
bleu_scores = []
for i in range(len(tokens["preds"])):
    p = tokens["preds"][i].replace(" ", "").lower()
    t = tokens["targs"][i].replace(" ", "").lower()
    print(f"p: {p} \t t: {t}")
    print()
    
    # Calculate ROUGE-L and BLEU scores
    rouge_l_scores = calculate_rouge_l(p, t)
    f_measures_L.append(rouge_l_scores['rougeL'].fmeasure)

    bleu_score = calculate_bleu_score(p, t)
    bleu_scores.append(bleu_score)

avg_l = sum(f_measures_L) / len(tokens["preds"])
print(f'Avg Rogue-L fmeasure: {avg_l}')
avg_bleu = sum(bleu_scores) / len(tokens["preds"])
print(f'Avg Bleu Score: {avg_bleu}')


