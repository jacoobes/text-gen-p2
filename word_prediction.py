#npx degit jghawaly/CSC7809_FoundationModels/Project2/data/raw raw/                             
import torch
from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
import sentencepiece as spm
from torch.optim import AdamW, lr_scheduler
from torcheval.metrics.text import Perplexity, BLEUScore
import json
from models import LSTM, RNNModel



def add_special_token(prompt, completion):
    # If the beginning of the prompt is upper case, then we assume it is the start of a sequence
    if prompt[0].isupper():
        prompt = '<bos>' + prompt
    # If the end of the completion is a terminating punctuation, then we assume it is the end of a sequence
    if completion.endswith('.') or completion.endswith('?') or completion.endswith('!'):
        completion += '<eos>'
    return  prompt, completion
 

class TokenizedDataset(Dataset):
    def __init__(self, jsonl_file, tokenizer, seq_length):
        self.data = []
        self.seq_length = seq_length

        # Load and pad sequences
        for line in jsonl_file:
            prompt  = line["prompt"]
            completion = line['completion']
            text = prompt + ' ' + completion
            token_ids = tokenizer.encode(text, out_type=int)[:seq_length]
            if len(token_ids) < 2:
                continue
            self.data.append(token_ids)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        tids = self.data[idx]
        #print(self.data[idx][0], self.data[idx][1])
        inp = torch.tensor(tids[:-1], dtype=torch.long)
        target = torch.tensor(tids[1:], dtype=torch.long)
        return inp, target


def read_jsonl(pat):
    with open(pat, 'r') as f:
        data = [json.loads(line) for line in f]
    return data



def training_kit(params, lr, weight_decay, dataloader, valloader, batch_size, epochs):
    opt = AdamW(params=params, lr=lr,weight_decay=weight_decay)
    return {
        # ignore padding.
        'opt': opt,
        'scheduler': lr_scheduler.ReduceLROnPlateau(opt, 'min', patience=3, factor=0.5),
        'epochs' : epochs,
        'batch_size': batch_size,
        'train_loader': dataloader,
        'val_loader': valloader
    }

def mkcollation(pad_id):
    def collate(batch):
        input_batch, target_batch = zip(*batch)
        input_batch = torch.nn.utils.rnn.pad_sequence(input_batch, batch_first=True, padding_value=pad_id)
        target_batch = torch.nn.utils.rnn.pad_sequence(target_batch, batch_first=True, padding_value=pad_id)
        return input_batch, target_batch
    return collate


if __name__ == '__main__':
    tokenizer_location = "bptokenizer.model"
    training_data = read_jsonl('./data/train.jsonl')
    testing_data = read_jsonl('./data/test.jsonl')
    sp = spm.SentencePieceProcessor(model_file='./bptokenizer.model')
    sp.LoadFromFile('./bptokenizer.model')
    # Some arbitrary parameters for the example
    hidden_size = 512  # Number of hidden units
    output_size = sp.GetPieceSize() # Output dimension
    seq_len = 30  # Length of the input sequence
    batch_size = 256  # Number of sequences in a batch
    embed_dim = 1024
    pad_id = sp.pad_id()
    print('pad_id', pad_id)
    collate = mkcollation(pad_id) 
    training_loader = DataLoader(
        TokenizedDataset(training_data, sp, seq_len),
        batch_size=batch_size,
        drop_last=True,
        shuffle=True,
        collate_fn=collate
    ) 
    valset, testset = torch.utils.data.random_split(TokenizedDataset(testing_data, sp, seq_len), [.8, .2])
    validation_loader = DataLoader(
        valset,
        shuffle=False,
        drop_last=True,
        batch_size=batch_size,
        collate_fn=collate
    )
    test_loader = DataLoader(
        testset,
        shuffle=False,
        drop_last=True,
        batch_size=batch_size,
        collate_fn=collate
    )
    if torch.cuda.is_available():
        print('torch cuda is_available')
        device = torch.device('cuda')          # Use GPU
    else:
        print('torch cuda not is_available')
        device = torch.device('cpu')           # Use CPU
#    device = torch.device('cpu')

    import argparse
    parser = argparse.ArgumentParser(description="CLI for word prediction model.")
    parser.add_argument("model", choices=["rnn", "transformer", "lstm"], help="Model type to use.")
    parser.add_argument("mode", choices=["train", "test"], help="Operation mode.")
    parser.add_argument("--state", type=str)
    
    args = parser.parse_args()
    
    print(f"Selected model: {args.model}")
    print(f"Mode: {args.mode}")
    if args.model == 'rnn':
        model = RNNModel(embed_dim=embed_dim,
                         hidden_size=hidden_size,
                         output_size=output_size,
                         batch_size=batch_size,
                         n_layers=3,
                         device=device,
                         tokenizer=sp,
                         name="rnn").to(device)
        trainkit = training_kit(params=model.parameters(),
                                lr=0.0001,
                                epochs=30,
                                weight_decay=0.01,
                                dataloader=training_loader,
                                valloader=validation_loader,
                                batch_size=batch_size)
    elif args.model == 'lstm':
        model = LSTM(embed_dim=embed_dim,
                     hidden_size=embed_dim,
                     output_size=output_size,
                     batch_size=batch_size,
                     n_layers=3,
                     device=device,
                     tokenizer=sp,
                     tie_weights=True,
                     name="lstm").to(device)
        trainkit = training_kit(params=model.parameters(),
                                lr=0.0001,
                                epochs=30,
                                weight_decay=0.01,
                                dataloader=training_loader,
                                valloader=validation_loader,
                                batch_size=batch_size)

    elif args.model == 'transformer':
        raise Exception('not impl')       
    else:
        raise Exception('unknown model type')

    if args.mode == 'train':
        model.reps(trainkit)
    else:
        metrics = {
            'perp': Perplexity(ignore_index=sp.pad_id()).to(device),
            'bleu': BLEUScore(n_gram=3).to(device)
        }
        model.eval()

        def evaluate_perplexity(model, perplexity_metric, data_loader, device):
            hidden = model.init_hidden(model.batch_size)
            with torch.no_grad():
                for inputs, labels in data_loader:
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    # Initialize hidden state for this batch
                    # Forward pass through the model
                    logits, hidden = model(inputs, hidden)
                    perplexity_metric.update(logits, labels)
                    #print(perplexity_metric.compute().item())
            
            # Compute perplexity: torcheval.Perplexity returns exp(avg_loss)
            ppl = perplexity_metric.compute().item()
            return ppl

        def evaluate_bleu(model, bleu_metric, data_loader, device):
             with torch.no_grad():
                for inputs, labels in data_loader:
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    # Initialize hidden state for this batch
                    # Forward pass through the model
                    logits, hidden = model(inputs, None)
                    bleu_metric.update(logits, labels)
           

        model.load_state_dict(torch.load(args.state))
        ppl = evaluate_perplexity(model, metrics['perp'], test_loader, device)
        print("perplexity", ppl)

        #bleu = evaluate_perplexity(model, metrics['bleu'], test_loader, device) print("bleu", bleu)

        print(model.prompt('Alice'))
    

