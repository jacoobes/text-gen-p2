import io
#npx degit jghawaly/CSC7809_FoundationModels/Project2/data/raw raw/                             
import os, glob
import urllib.request
import numpy as np
import torch
import numpy as np
from functools import reduce
import sentencepiece as spm
from torch.autograd import Variable
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
from torcheval.metrics.text import Perplexity, BLEUScore
import json
from datetime import datetime

from models import RNNModel



def train_seq_model(model, train_kit):
    loss_fn = train_kit['loss']
    # Initializing in a separate cell so we can easily add more epochs to the same run
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    epoch_number = 0
    EPOCHS = train_kit['epochs']
    best_vloss = 1_000_000.
    hidden = model.init_hidden(train_kit['batch_size'])   

    for epoch in range(EPOCHS):
        print('EPOCH {}:'.format(epoch + 1))

        # Make sure gradient tracking is on, and do a pass over the data
        model.train(True)

        
        optimizer = train_kit['opt']
        training_loader = train_kit['loader']
        loss_fn = train_kit['loss']


        running_loss = 0.
        last_loss = 0.

        # Here, we use enumerate(training_loader) instead of
        # iter(training_loader) so that we can track the batch
        # index and do some intra-epoch reporting
        for i, data in enumerate(training_loader):
            # Every data instance is an input + label pair
            inputs, labels = data
            hidden = tuple([Variable(each.data) for each in hidden])
            # Zero your gradients for every batch!
            optimizer.zero_grad()
            
            outputs, hidden = model.forward(inputs, hidden)

            loss = loss_fn(outputs, labels.view(-1))
            print("loss backward")
            loss.backward()
            print("optimizer")
            # Adjust learning weights
            optimizer.step()

            # Gather data and report
            running_loss += loss.item()
            if i % 1000 == 999:
                last_loss = running_loss / 1000 # loss per batch
                print('  batch {} loss: {}'.format(i + 1, last_loss))
                tb_x = epoch * len(training_loader) + i + 1
                print('Loss/train' , last_loss, tb_x)
                running_loss = 0.




        running_vloss = 0.0
        # Set the model to evaluation mode, disabling dropout and using population
        # statistics for batch normalization.
        model.eval()

        # Disable gradient computation and reduce memory consumption.
        with torch.no_grad():
            for i, vdata in enumerate(validation_loader):
                vinputs, vlabels = vdata
                voutputs = model(vinputs)
                vloss = loss_fn(voutputs, vlabels)
                running_vloss += vloss

        avg_vloss = running_vloss / (i + 1)
        print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))

        # Log the running loss averaged per batch
        # for both training and validation
        print('Training vs. Validation Loss',
                        { 'Training' : avg_loss, 'Validation' : avg_vloss },
                        epoch_number + 1)
        #writer.flush()

        # Track best performance, and save the model's state
        if avg_vloss < best_vloss:
            best_vloss = avg_vloss
            model_path = 'model_{}_{}'.format(timestamp, epoch_number)
            torch.save(model.state_dict(), model_path)

        epoch_number += 1

import torch
from torch.utils.data import Dataset, DataLoader

class TokenizedDataset(Dataset):
    def __init__(self, jsonl_file, tokenizer, seq_length):
        self.data = []
        self.seq_length = seq_length

        # Load and pad sequences
        for line in jsonl_file:
            tokens = tokenizer.Encode(line["prompt"])
            completion = tokenizer.Encode(line["completion"])
            if len(tokens) > seq_length:
                tokens = tokens[:seq_length]
            else:
                tokens += [0] * (seq_length - len(tokens))  # Pad with 0s
            self.data.append([tokens, [completion[0]]])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        #print(self.data[idx][0], self.data[idx][1])
        return torch.tensor(self.data[idx][0], dtype=torch.long), \
               torch.tensor(self.data[idx][1], dtype=torch.long)


def read_jsonl(pat):
    with open(pat, 'r') as f:
        data = [json.loads(line) for line in f]
    return data

def train_bpe(response, output: str):
    model = io.BytesIO()
    spm.SentencePieceTrainer.Train(
      input=response, 
      model_writer=model,
      vocab_size=10000,
      model_type="bpe",
      bos_id=1,
      eos_id=2,
      pad_id=3,
      user_defined_symbols=",".join(['<bos>', '<eos>', '<pad>' ])
    )
      
    # Serialize the model as file.
    with open(output, 'wb') as f:
       f.write(model.getvalue())

def training_kit(params, lr, weight_decay, dataloader, batch_size):
    return {
        'loss': CrossEntropyLoss(),
        'opt': AdamW(params=params, lr=lr,weight_decay=weight_decay),
        'epochs' : 30,
        'batch_size': batch_size,
        'loader': dataloader
    }


if __name__ == '__main__':
    tokenizer_location = "model.bpe"
    training_data = read_jsonl('./train.jsonl')

    if not os.path.exists(tokenizer_location):
        corpus = ""
        for file in glob.glob("*.txt", root_dir="./raw"):
            with open('raw/'+file, 'r', encoding="utf8") as f:
                corpus +=  f.read() + " "
        with open('./corpus.txt', 'w+', encoding='utf8') as cr:
            cr.write(corpus)

        train_bpe('corpus.txt', output=tokenizer_location)

    sp = spm.SentencePieceProcessor(tokenizer_location)
    # Some arbitrary parameters for the example
    input_size = 256  # Number of input features
    hidden_size = 64  # Number of hidden units
    output_size = sp.vocab_size() # Output dimension
    seq_len = 50  # Length of the input sequence
    batch_size = 256  # Number of sequences in a batch
    embed_dim = 128
    dataloader = DataLoader(
        TokenizedDataset(training_data, sp, seq_len),
        batch_size=batch_size,
        shuffle=True
    ) 

    model = RNNModel(embed_dim=embed_dim,
                     hidden_size=hidden_size,
                     output_size=output_size,
                     n_layers=2,
                     batch_size=batch_size) 

    trainkit = training_kit(params=model.parameters(),
                            lr=0.001,
                            weight_decay=0.01,
                            dataloader=dataloader,
                            batch_size=batch_size)

    train_seq_model(model, train_kit=trainkit)
