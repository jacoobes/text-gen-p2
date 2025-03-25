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
    def __init__(self, output_size, hidden_size, n_layers, batch_size, embed_dim):
        super(RNNModel, self).__init__()
        self.emb = nn.Embedding(output_size, embed_dim)
        self.rnn = nn.RNN(embed_dim, hidden_size, num_layers=n_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.n_layers=n_layers
        self.hidden_size=hidden_size
        self.output_size=output_size
        self.batch_size=batch_size
        

    def prompt(self, input, hidden):
        # input shape: [seq_len, batch_size]
        logits, hidden = self.forward(input, hidden)
        # logits shape: [seq_len * batch_size, vocab_size]
        # hidden shape: [2, num_layers, batch_size, rnn_size]
        probs = nn.functional.softmax(logits)
        # shape: [seq_len * batch_size, vocab_size]
        probs = probs.view(input.size(0), input.size(1), probs.size(1))
        # output shape: [seq_len, batch_size, vocab_size]
        return probs, hidden
        

    def forward(self, x, hidden):

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
    def __init__(self):
        ...

    def prompt(self, s: str):
        ...

    def forward(self):
        ...

