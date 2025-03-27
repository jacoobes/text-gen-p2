import io
#npx degit jghawaly/CSC7809_FoundationModels/Project2/data/raw raw/                             
import os, glob
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple
import numpy as np
import torch
import numpy as np
from functools import reduce
import sentencepiece as spm
from torch.autograd import Variable
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW, lr_scheduler
from torcheval.metrics.metric import Metric
from torcheval.metrics.text import Perplexity, BLEUScore
import json
from tqdm import tqdm
from datetime import datetime

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
        'loss': CrossEntropyLoss(ignore_index=3),
        'opt': opt,
        'scheduler': lr_scheduler.ReduceLROnPlateau(opt, 'min', patience=1, factor=0.5),
        'epochs' : epochs,
        'batch_size': batch_size,
        'train_loader': dataloader,
        'val_loader': valloader
    }

def collation(batch):
    input_batch, target_batch = zip(*batch)
    input_batch = torch.nn.utils.rnn.pad_sequence(input_batch, batch_first=True, padding_value=3)
    target_batch = torch.nn.utils.rnn.pad_sequence(target_batch, batch_first=True, padding_value=3)
    return input_batch, target_batch

def measure(metric: Metric, input, output):
    metric.update(input, output)
    return metric.compute()


if __name__ == '__main__':
    tokenizer_location = "bptokenizer.model"
    training_data = read_jsonl('./data/train.jsonl')
    testing_data = read_jsonl('./data/test.jsonl')
    sp = spm.SentencePieceProcessor(model_file=tokenizer_location)

    # Some arbitrary parameters for the example
    hidden_size = 52  # Number of hidden units
    output_size = sp.GetPieceSize() # Output dimension
    seq_len = 50  # Length of the input sequence
    batch_size = 128  # Number of sequences in a batch
    embed_dim = 512
     
    training_loader = DataLoader(
        TokenizedDataset(training_data, sp, seq_len),
        batch_size=batch_size,
        drop_last=True,
        shuffle=True,
        collate_fn=collation
    ) 
    valset, testset = torch.utils.data.random_split(TokenizedDataset(testing_data, sp, seq_len), [.8, .2])
    validation_loader = DataLoader(
        valset,
        shuffle=False,
        drop_last=True,
        batch_size=batch_size,
        collate_fn=collation
    )
    test_loader = DataLoader(
        testset,
        shuffle=False,
        drop_last=True,
        batch_size=batch_size,
        collate_fn=collation
    )
    device = 'cpu'
    rnnmodel = RNNModel(embed_dim=embed_dim,
                        hidden_size=hidden_size,
                        output_size=output_size,
                        batch_size=batch_size,
                        n_layers=4,
                        device=device,
                        tokenizer=sp,
                        name="rnn")

    lstmmodel = LSTM(embed_dim=embed_dim,
                     hidden_size=hidden_size,
                     output_size=output_size,
                     batch_size=batch_size,
                     n_layers=4,
                     device=device,
                     tokenizer=sp,
                     name="lstm")

    trainkit = training_kit(params=lstmmodel.parameters(),
                            lr=0.0001,
                            epochs=30,
                            weight_decay=0.01,
                            dataloader=training_loader,
                            valloader=validation_loader,
                            batch_size=batch_size)

    # lstmmodel.reps(trainkit)

    metrics = {
        'perp': Perplexity(ignore_index=3),
        'bleu': BLEUScore(n_gram=2)
    }
    lstmmodel.eval()

    def evaluate_perplexity(model, perplexity_metric, data_loader, device):
        model.eval()
        # Initialize the Perplexity metric from torcheval.
        perplexity_metric = Perplexity(ignore_index=3)
        
        with torch.no_grad():
            for inputs, labels in data_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                # Initialize hidden state for this batch
                # Forward pass through the model
                logits, _ = model(inputs, None)
                perplexity_metric.update(logits, labels)
                #print(perplexity_metric.compute().item())
        
        # Compute perplexity: torcheval.Perplexity returns exp(avg_loss)
        ppl = perplexity_metric.compute().item()
        return ppl

    lstmmodel.load_state_dict(torch.load('./model_20250327_154622_lstm.torch'))

    ppl = evaluate_perplexity(lstmmodel, metrics['perp'], test_loader, device)
    print("perplexity", ppl)
    print(lstmmodel.prompt('The wizard'))
    

