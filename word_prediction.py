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
 
   

def train_seq_model(model, train_kit):
    loss_fn = train_kit['loss']
    optimizer = train_kit['opt']
    training_loader = train_kit['train_loader']

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    epoch_number = 0
    EPOCHS = train_kit['epochs']
    best_vloss = 1_000_000.
    training_loss, val_loss = [], []
    for epoch in range(EPOCHS):
        # Make sure gradient tracking is on, and do a pass over the data
        model.train(True)
        running_loss = 0.
        for inputs, labels in training_loader:

            # Zero your gradients for every batch!
            optimizer.zero_grad()
            # print(labels[0])
            logits, _ = model.forward(inputs)
            loss = loss_fn(logits.view(-1, model.output_size), labels.view(-1))
            #print("loss backward")
            loss.backward()
            #print("optimizer")

            # Adjust learning weights
            optimizer.step()

            # Gather data and report
            running_loss += loss.item()
            print(f"EPOCH {epoch} Loss: {loss.item()}")

        avg_train_loss = running_loss / len(training_loader)
        training_loss.append(avg_train_loss)

        running_vloss = 0.0
        # Set the model to evaluation mode, disabling dropout and using population
        # statistics for batch normalization.
        model.eval()

        # Disable gradient computation and reduce memory consumption.
        with torch.no_grad():
            for vinputs, vlabels  in train_kit['val_loader']: 
                voutputs, hidden = model(vinputs)
                vloss = loss_fn(voutputs.view(-1, model.output_size), vlabels.view(-1))
                running_vloss += vloss

        avg_vloss = running_vloss / len(train_kit['val_loader'])
        val_loss.append(avg_vloss)
        print('loss {} val {}'.format(avg_train_loss, avg_vloss))

        # Track best performance, and save the model's state
        if avg_vloss < best_vloss:
            best_vloss = avg_vloss

        train_kit['scheduler'].step(best_vloss)
        epoch_number += 1

    model_path = 'model_{}_{}.torch'.format(timestamp, model.name)
    torch.save(model.state_dict(), model_path)
    return training_loss, val_loss



class TokenizedDataset(Dataset):
    def __init__(self, jsonl_file, tokenizer, seq_length):
        self.data = []
        self.seq_length = seq_length

        # Load and pad sequences
        for line in jsonl_file:
            prompt  = line["prompt"]
            completion = line['completion']
            prompt, completion = add_special_token(prompt, completion)
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



def training_kit(params, lr, weight_decay, dataloader, valloader, batch_size):
    opt = AdamW(params=params, lr=lr,weight_decay=weight_decay)
    return {
        # ignore padding.
        'loss': CrossEntropyLoss(ignore_index=3),
        'opt': opt,
        'scheduler': lr_scheduler.ReduceLROnPlateau(opt, 'min', patience=1, factor=0.5),
        'epochs' : 3,
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
    sp = spm.SentencePieceProcessor(tokenizer_location)

    # Some arbitrary parameters for the example
    hidden_size = 24  # Number of hidden units
    output_size = sp.vocab_size() # Output dimension
    seq_len = 48  # Length of the input sequence
    batch_size = 256  # Number of sequences in a batch
    embed_dim = 128
     
    training_loader = DataLoader(
        TokenizedDataset(training_data, sp, seq_len),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collation
    ) 
    valset, trainset = torch.utils.data.random_split(TokenizedDataset(testing_data, sp, seq_len), [.8, .2])
    validation_loader = DataLoader(
        valset,
        shuffle=False,
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
                            weight_decay=0.01,
                            dataloader=training_loader,
                            valloader=validation_loader,
                            batch_size=batch_size)

    #train_seq_model(lstmmodel, train_kit=trainkit)

    metrics = {
        'perp': Perplexity(ignore_index=3),
        'bleu': BLEUScore(n_gram=2)
    }
    lstmmodel.load_state_dict(torch.load('./model_20250326_211255_lstm.torch'))
    lstmmodel.eval()
    print(lstmmodel.prompt('The wizard of waverly place'))

    

