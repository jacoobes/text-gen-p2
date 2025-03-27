from torch import nn
import torch
from torch.autograd import Variable

class Transformer(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, embed_dim):
        ...

    def prompt(self, s: str):
        ...

    def forward(self):
        ...

class RNNModel(nn.Module):
    def __init__(self, 
                output_size, 
                hidden_size,
                n_layers,
                batch_size,
                embed_dim,
                tokenizer,
                device,
                name='rnn',
                pad_token_id=0,
                dropout=0.2,):
        super(RNNModel, self).__init__()

        # vocab size 
        self.emb = nn.Embedding(output_size, 
                                embed_dim,
                                padding_idx=pad_token_id)
        self.rnn = nn.RNN(input_size=embed_dim, 
                          hidden_size=hidden_size,
                          num_layers=n_layers,
                          batch_first=True,
                          dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)
        self.tokenizer=tokenizer
        self.n_layers=n_layers
        self.hidden_size=hidden_size
        self.output_size=output_size
        self.batch_size=batch_size
        self.device=device
        self.name=name
        

    def prompt(self, text: str, max_length = 50):
        self.eval()
        # encode, turn to tensor, and turn dimensions into batchable dimensions
        input_tensor = torch.tensor(self.tokenizer.encode(text), dtype=torch.long).unsqueeze(0)
        hidden = None
        output = []
        for _ in range(max_length):
            with torch.no_grad():
                logits, hidden = self.forward(input_tensor, hidden)
                logits = logits.squeeze(0)
                probs = nn.functional.softmax(logits, dim=0)
                next_token = torch.argmax(probs, dim=-1)
                # todo, fix
                next_token_id = next_token.item()
                output.append(next_token_id)
                input_tensor = torch.tensor([[ next_token_id ]], dtype=torch.long )

                
        return self.tokenizer.decode(output, out_type=str)
        

    def forward(self, x, hidden=None):

        x = self.emb(x)
        rnn_output, hidden = self.rnn(x, hidden)
        out = self.fc(rnn_output)
        return out, hidden

    def init_hidden(self, batch_size):
        # create 2 new zero tensors of size n_layers * batch_size * hidden_dim
        hidden = (Variable(torch.zeros(self.rnn.num_layers, batch_size, self.hidden_size)),
                  Variable(torch.zeros(self.rnn.num_layers, batch_size, self.hidden_size)))
        # initialize hidden state with zero weights, and move to GPU if available
        
        return hidden
        


class LSTM(nn.Module):
    def __init__(self,                 
            output_size, 
            hidden_size,
            n_layers,
            batch_size,
            embed_dim,
            tokenizer,
            device,
            name='lstm',
            pad_token_id=0,
            dropout=0.2):
        super(LSTM, self).__init__()
        self.emb = nn.Embedding(output_size, 
                                embed_dim,
                                padding_idx=pad_token_id)
        self.lstm= nn.LSTM(input_size=embed_dim, 
                          hidden_size=hidden_size,
                          num_layers=n_layers,
                          batch_first=True,
                          dropout=dropout)
        self.tokenizer=tokenizer
        self.fc = nn.Linear(hidden_size, output_size)
        self.n_layers=n_layers
        self.hidden_size=hidden_size
        self.output_size=output_size
        self.batch_size=batch_size
        self.device=device
        self.name=name

    def prompt(self, text: str, max_length=50):
        self.eval()
        # encode, turn to tensor, and turn dimensions into batchable dimensions
        input_tensor = torch.tensor(self.tokenizer.encode(text), dtype=torch.long).unsqueeze(0)
        hidden = None
        output = []
        for _ in range(max_length):
            with torch.no_grad():
                logits, hidden = self.forward(input_tensor, hidden)
                print(logits.shape, input_tensor.shape)
                logits = logits[:,-1, :]
                probs = nn.functional.softmax(logits, dim=1)
                print(probs)
                next_token = torch.argmax(probs, dim=-1) # greedy sampling
                print(next_token)
                # todo, fix
                next_token_id = next_token.item()
                output.append(next_token_id)
                input_tensor = torch.tensor([[ next_token_id ]], dtype=torch.long )

                
        return self.tokenizer.decode(output, out_type=str)


    def forward(self, x, hidden=None):
        x = self.emb(x)
        lstm_output, hidden = self.lstm(x, hidden)
        out = self.fc(lstm_output)
        return out, hidden

