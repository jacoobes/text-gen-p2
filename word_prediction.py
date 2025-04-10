#npx degit jghawaly/CSC7809_FoundationModels/Project2/data/raw raw/                             
import torch
from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
from torchview import draw_graph
import sentencepiece as spm
from torch.optim import AdamW, lr_scheduler
from torcheval.metrics.text import Perplexity, BLEUScore 
import json
from models import LSTM, RNNModel, TransformerModel



def add_special_token(prompt, completion):
    # If the beginning of the prompt is upper case, then we assume it is the start of a sequence
    if prompt[0].isupper():
        prompt = '<bos>' + prompt
    # If the end of the completion is a terminating punctuation, then we assume it is the end of a sequence
    if completion.endswith('.') or completion.endswith('?') or completion.endswith('!'):
        completion += '<eos>'
    return  prompt, completion

class PromptCompletionDataset(Dataset):
    def __init__(self, jsonl_file, tokenizer, seq_length):
        self.data = []
        self.seq_length = seq_length

        # Load and pad sequences
        for line in jsonl_file:
            prompt  = line["prompt"]
            completion = line['completion']
            text = (prompt + ' ' + completion)[:seq_length]
            token_ids = tokenizer.encode(text, out_type=int)[:seq_length]
            if len(text) < 2:
                continue
            self.data.append(tokenizer.decode(token_ids, out_type=str))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        tids = self.data[idx]
        #print(self.data[idx][0], self.data[idx][1])
        inp = tids[:-1]
        target = tids[1:]
        return inp, target

   

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

def evaluate_perplexity(model, perplexity_metric, data_loader, device):
    if isinstance(model, TransformerModel):
        hidden = None
    else:
        hidden = model.init_hidden(model.batch_size)
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            # Initialize hidden state for this batch
            # Forward pass through the model
            if not isinstance(model, TransformerModel):
                logits, hidden = model(inputs, hidden)
            else:
                logits = model(inputs, labels)

            perplexity_metric.update(logits, labels)
            #print(perplexity_metric.compute().item())
    
    # Compute perplexity: torcheval.Perplexity returns exp(avg_loss)
    ppl = perplexity_metric.compute().item()
    return ppl


def evaluate_bleu(model, bleu_metric, data_loader):
    model.eval()
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = inputs[0]
            labels = labels[0]

            prediction = model.prompt(inputs, argm=False)
            bleu_metric.update([prediction], [[labels]])
    
    bleu_score = bleu_metric.compute().item()
    return bleu_score

if __name__ == '__main__':
    tokenizer_location = "bptokenizer.model"
    training_data = read_jsonl('./data/train.jsonl')
    testing_data = read_jsonl('./data/test.jsonl')
    sp = spm.SentencePieceProcessor(model_file='./bptokenizer.model')
    # Some arbitrary parameters for the example
    seq_len = 30  # Length of the input sequence
    batch_size = 256  # Number of sequences in a batch
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

    import argparse
    parser = argparse.ArgumentParser(description="CLI for word prediction model.")
    parser.add_argument("model", choices=["rnn", "transformer", "lstm"], help="Model type to use.")
    parser.add_argument("mode", choices=["train", "test", 'draw'], help="Operation mode.")
    parser.add_argument("--state", type=str)
    
    args = parser.parse_args()
    
    print(f"Selected model: {args.model}")
    print(f"Mode: {args.mode}")
    if args.model == 'rnn':
        hidden_size = 128  # Number of hidden units
        output_size = sp.GetPieceSize() # Output dimension
        seq_len = 30  # Length of the input sequence
        batch_size = 256  # Number of sequences in a batch
        embed_dim = 1024

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
        def mg():
            return draw_graph(model, input_data=(torch.randint(size=(batch_size, seq_len), low=0, high=400),
                                                 model.init_hidden(model.batch_size)),
                                     graph_name=args.model,
                                     roll=True,
                                     filename="./good/"+args.model,
                                     save_graph=True,
                                     device='meta'
                                ),

    elif args.model == 'lstm':
        hidden_size = 512  # Number of hidden units
        output_size = sp.GetPieceSize() # Output dimension
        seq_len = 30  # Length of the input sequence
        batch_size = 256  # Number of sequences in a batch
        embed_dim = 1024

        model = LSTM(embed_dim=embed_dim,
                     hidden_size=hidden_size,
                     output_size=output_size,
                     batch_size=batch_size,
                     n_layers=4,
                     device=device,
                     tokenizer=sp,
                     name="lstm").to(device)
        trainkit = training_kit(params=model.parameters(),
                                lr=0.0001,
                                epochs=30,
                                weight_decay=0.01,
                                dataloader=training_loader,
                                valloader=validation_loader,
                                batch_size=batch_size)
        def mg():
            return draw_graph(model, 
                             graph_name=args.model,
                             device='meta',
                             input_data=(torch.randint(size=(batch_size, seq_len), low=0, high=400),
                                         model.init_hidden(model.batch_size)),
                             roll=True,
                             filename="./good/"+args.model,
                             save_graph=True)


    elif args.model == 'transformer':
        embed_dim=320
        output_size=sp.GetPieceSize()
        feedforward_size=1024
        batch_size=256
        seq_len = 50  # Length of the input sequence
        model = TransformerModel(
                    feedforward_size=feedforward_size,
                    embed_dim=embed_dim,
                    output_size=output_size,
                    batch_size=batch_size,
                    sequence_length=seq_len,
                    device=device,
                    tokenizer=sp,
                    name="transformer"
                ).to(device)
        def mg():
            return draw_graph(model, 
                                 graph_name=args.model,
                                 device='meta',
                                 input_data=(torch.randint(size=(batch_size, seq_len), low=0, high=10000)),
                                 roll=True,
                                 filename="./good/"+args.model,
                                 save_graph=True)
        trainkit = training_kit(params=model.parameters(),
                                lr=0.0001,
                                epochs=75,
                                weight_decay=0.01,
                                dataloader=training_loader,
                                valloader=validation_loader,
                                batch_size=batch_size)

    else:
        raise Exception('unknown model type')

    if args.mode == 'train':
        model.reps(trainkit)
    elif args.mode == 'test':
        metrics = {
            'perp': Perplexity(ignore_index=sp.pad_id()).to(device),
            'bleu': BLEUScore(n_gram=2).to(device)
        }
        model.eval()



        model.load_state_dict(torch.load(args.state))
        ppl = evaluate_perplexity(model, metrics['perp'], test_loader, device)
        print("perplexity", ppl)

#        bleu = evaluate_bleu(model, metrics['bleu'], DataLoader(PromptCompletionDataset(testing_data, sp, seq_len), batch_size=1)) 
#        print("bleu", bleu)

        print(model.prompt('Who is Alice?'))
    else:
        cg = mg()
        print(cg)
