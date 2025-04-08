from torch import nn
import json

import math
import torch
from datetime import datetime
from tqdm import tqdm

def sampler(top_k:int, logits: torch.Tensor, top_p=0.0, filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k >0: keep only top k tokens with highest probability (top-k filtering).
            top_p >0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        
        Basic outline taken from https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    assert logits.dim() == 2  # [BATCH_SIZE, VOCAB_SIZE]
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k, dim=1)[0][..., -1, None]
        logits[indices_to_remove] = filter_value
    
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    
    cumulative_probs = torch.cumsum(torch.nn.functional.softmax(sorted_logits, dim=-1), dim=-1)
    # Remove tokens with cumulative probability above the threshold
    sorted_indices_to_remove = cumulative_probs > top_p
    # Shift the indices to the right to keep also the first token above the threshold
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0
    
    # Replace logits to be removed with -inf in the sorted_logits
    sorted_logits[sorted_indices_to_remove] = filter_value
    # Then reverse the sorting process by mapping back sorted_logits to their original position
    logits = torch.gather(sorted_logits, 1, sorted_indices.argsort(-1))
    
    pred_token = torch.multinomial(torch.nn.functional.softmax(logits, -1), 1) # [BATCH_SIZE, 1]
    return pred_token





class PositionalEncoding(nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens in the sequence.
        The positional encodings have the same dimension as the embeddings, so that the two can be summed.
        Here, we use sine and cosine functions of different frequencies.
    .. math:
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
    Examples:
        >>> pos_encoder = PositionalEncoding(d_model)
    """

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        Examples:
            >>> output = pos_encoder(x)
        """

        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class TransformerModel(nn.Module):
    def __init__(self, 
        output_size,
        batch_size,
        feedforward_size,
        sequence_length,
        embed_dim,
        tokenizer,
        device,
        max_seq_len=50,
        name='transformer'):
        super(TransformerModel, self).__init__()
        self.max_seq_len = max_seq_len
        self.batch_size=batch_size
        self.seqlen = sequence_length
        self.embedding = nn.Embedding(output_size, 
                                      embed_dim,
                                      padding_idx=tokenizer.pad_id())

        self.transformer = nn.Transformer(d_model=embed_dim, # embedding dimension
                                         nhead=8, # num of attention heads
                                         dim_feedforward=feedforward_size,
                                         batch_first=True)
        self.posenc = PositionalEncoding(d_model=embed_dim)
        self.criterion  = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_id())
        self.fc = nn.Linear(embed_dim, output_size)
        self.logsoft = nn.LogSoftmax(dim=-1)
        self.output_size=output_size
        self.device=device
        self.name=name
        self.tokenizer=tokenizer

    def reps(self, train_kit):
        optimizer = train_kit['opt']
        training_loader = train_kit['train_loader']
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        epochs = train_kit['epochs']
        patience_counter = 0  # Track epochs without improvement
        patience=5
        best_vloss = 1_000_000.
        training_loss, val_loss = [], []
        for epoch in tqdm(range(epochs)):
            # Make sure gradient tracking is on, and do a pass over the data
            self.train()
            running_loss = 0.
            for inputs, labels in training_loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                # Zero your gradients for every batch!
                optimizer.zero_grad()
                logits= self.forward(inputs, labels)
                # logits shape should be (batch_size, seqlen, vocabsize)
                # labels should be (batch_size, expected_seq)
                #print(logits.shape)
                #print(labels.shape)
                loss = self.criterion(logits.view(-1, self.output_size), labels.view(-1))
                #print("loss backward")
                loss.backward()
                #print("optimizer")
                torch.nn.utils.clip_grad_norm_(self.parameters(), 3)
                # Adjust learning weights
                optimizer.step()

                # Gather data and report
                running_loss += loss.item()
                #print(f"EPOCH {epoch+1} Loss: {loss.item()}")

            avg_train_loss = running_loss / len(training_loader)
            training_loss.append(avg_train_loss)

            # Set the model to evaluation mode, disabling dropout and using population
            # statistics for batch normalization.
            self.eval()
            running_vloss = 0.0

            # Disable gradient computation and reduce memory consumption.
            with torch.no_grad():
                for vinputs, vlabels  in train_kit['val_loader']: 
                    vinputs, vlabels = vinputs.to(self.device), vlabels.to(self.device)
                    voutputs = self.forward(vinputs, vlabels)
                    vloss = self.criterion(voutputs.view(-1, self.output_size), vlabels.view(-1))
                    running_vloss += vloss.item()
            avg_vloss = running_vloss / len(train_kit['val_loader'])
            val_loss.append(avg_vloss)
            print('numbers = (avg loss {} avgval {} bestval {})'.format(avg_train_loss, avg_vloss, best_vloss))

            # Track best performance, and save the model's state
            if avg_vloss < best_vloss:
                print('patience reset')
                best_vloss = avg_vloss
                patience_counter=0
            else:
                print('patience counted increased')
                patience_counter+=1

            if patience_counter >= patience:
                print('not patience anymore. early stopping')
                break

            train_kit['scheduler'].step(avg_vloss)

        model_path = 'model_{}_{}.torch'.format(timestamp, self.name)
        torch.save(self.state_dict(), model_path)
        with open('./model_{}_{}_{}.json'.format(timestamp, self.name, 'tvl'), 'w+') as f:
            json.dump([training_loss,val_loss], f)

        return training_loss, val_loss

    def prompt(self, text: str, max_length = 50, argm=True):
        self.eval()
        # encode, turn to tensor, and turn dimensions into batchable dimensions
        input_tensor = torch.tensor(self.tokenizer.encode(text), dtype=torch.long).unsqueeze(0).to(self.device)
        output = []

        with torch.no_grad():
            for _ in range(max_length):
                logits = self.forward(input_tensor)
                # dim = (batchsize, token seq len, vocab size)
                logits = logits[:,-1, :]
                #probs = torch.nn.functional.softmax(logits, dim=-1)
                #next_token_id = torch.multinomial(probs, 1).item()                
                if not argm: 
                    next_token_id = sampler(10, logits, top_p=0.8)
                else:
                    next_token_id = torch.argmax(logits ,dim=-1)

                if self.tokenizer.eos_id() == next_token_id:
                    print('early stopping')
                    break

                output.append(next_token_id.item())

                input_tensor = torch.tensor([[next_token_id]], dtype=torch.long).to(self.device)

                
        return self.tokenizer.decode(output, out_type=str)

    def forward(self, input, targt=None):
        # embed input
        x = self.posenc.forward(self.embedding(input))
        tgt = self.posenc.forward(self.embedding(targt))
        # encoder
        x = self.transformer.forward(x, tgt)
        # decode
        x = self.fc(x)
        # activate
        x = self.logsoft.forward(x)
        return x 

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
                dropout=0.5):
        super(RNNModel, self).__init__()

        self.criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_id() )
        # vocab size 
        self.emb = nn.Embedding(output_size, 
                                embed_dim,
                                padding_idx=tokenizer.pad_id())
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
        

    def prompt(self, text: str, max_length = 50, argm=True):
        self.eval()
        # encode, turn to tensor, and turn dimensions into batchable dimensions
        input_tensor = torch.tensor(self.tokenizer.encode(text), dtype=torch.long).unsqueeze(0).to(self.device)
        hidden = self.init_hidden(input_tensor.size(0))
        output = []

        with torch.no_grad():
            for _ in range(max_length):
                logits, hidden = self.forward(input_tensor, hidden)
                # dim = (batchsize, token seq len, vocab size)
                logits = logits[:,-1, :]

                #probs = torch.nn.functional.softmax(logits, dim=-1)
                #next_token_id = torch.multinomial(probs, 1).item()                
                if not argm: 
                    next_token_id = sampler(10, logits, top_p=0.8)
                else:
                    next_token_id = torch.argmax(logits ,dim=-1)

                if self.tokenizer.eos_id() == next_token_id:
                    print('early stopping')
                    break

                output.append(next_token_id.item())

                input_tensor = torch.tensor([[next_token_id]], dtype=torch.long).to(self.device)

                
        return self.tokenizer.decode(output, out_type=str)

        

    def forward(self, x, hidden=None):
        x = self.emb(x)
        rnn_output, hidden = self.rnn(x, hidden)
        out = self.fc(rnn_output)
        return out, hidden

    def init_hidden(self, batch_size):
        hidden = torch.zeros(self.rnn.num_layers, batch_size, self.hidden_size)
        # initialize hidden state with zero weights, and move to GPU if available
        return hidden.to(self.device)

    def reps(self, train_kit):
        optimizer = train_kit['opt']
        training_loader = train_kit['train_loader']
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        epochs = train_kit['epochs']
        patience_counter = 0  # Track epochs without improvement
        patience=5
        best_vloss = 1_000_000.

        training_loss, val_loss = [], []
        hidden = self.init_hidden(self.batch_size)
        for epoch in tqdm(range(epochs)):
            # Make sure gradient tracking is on, and do a pass over the data
            self.train()
            running_loss = 0.
            for inputs, labels in training_loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                # Zero your gradients for every batch!
                optimizer.zero_grad()
                hidden = hidden.detach()
                logits, hidden = self.forward(inputs, hidden)
                # logits shape should be (batch_size, seqlen, vocabsize)
                # labels should be (batch_size, expected_seq)
                #print(logits.shape)
                #print(labels.shape)
                loss = self.criterion(logits.view(-1, self.output_size), labels.view(-1))
                #print("loss backward")
                loss.backward()
                #print("optimizer")
                torch.nn.utils.clip_grad_norm_(self.parameters(), 1)
                # Adjust learning weights
                optimizer.step()

                # Gather data and report
                running_loss += loss.item()
                #print(f"EPOCH {epoch+1} Loss: {loss.item()}")

            avg_train_loss = running_loss / len(training_loader)
            training_loss.append(avg_train_loss)

            # Set the model to evaluation mode, disabling dropout and using population
            # statistics for batch normalization.
            self.eval()
            running_vloss = 0.0

            vhidden = self.init_hidden(self.batch_size)
            # Disable gradient computation and reduce memory consumption.
            with torch.no_grad():
                for vinputs, vlabels  in train_kit['val_loader']: 
                    vinputs, vlabels = vinputs.to(self.device), vlabels.to(self.device)
                    vhidden = vhidden.detach()
                    voutputs, vhidden = self.forward(vinputs, vhidden)
                    vloss = self.criterion(voutputs.view(-1, self.output_size), vlabels.view(-1))
                    running_vloss += vloss.item()
            avg_vloss = running_vloss / len(train_kit['val_loader'])
            val_loss.append(avg_vloss)
            print('numbers = (avg loss {} avgval {} bestval {})'.format(avg_train_loss, avg_vloss, best_vloss))

            # Track best performance, and save the model's state
            if avg_vloss < best_vloss:
                print('patience reset')
                best_vloss = avg_vloss
                patience_counter=0
            else:
                print('patience counted increased')
                patience_counter+=1

            if patience_counter >= patience:
                print('not patience anymore. early stopping')
                break

            train_kit['scheduler'].step(avg_vloss)

        model_path = 'model_{}_{}.torch'.format(timestamp, self.name)
        torch.save(self.state_dict(), model_path)
        with open('./model_{}_{}_{}.json'.format(timestamp, self.name, 'tvl'), 'w+') as f:
            json.dump([training_loss,val_loss], f)

        return training_loss, val_loss


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
            tie_weights=False,
            dropout=0.4):
        super(LSTM, self).__init__()
        self.emb = nn.Embedding(output_size, 
                                embed_dim,
                                padding_idx=tokenizer.pad_id())
        self.lstm= nn.LSTM(input_size=embed_dim, 
                          hidden_size=hidden_size,
                          num_layers=n_layers,
                          batch_first=True,
                          dropout=dropout)
        self.criterion =  nn.CrossEntropyLoss(ignore_index=tokenizer.pad_id())
        self.tokenizer=tokenizer
        self.fc = nn.Linear(hidden_size, output_size)
        self.n_layers=n_layers
        self.hidden_size=hidden_size
        self.output_size=output_size
        self.batch_size=batch_size
        self.device=device
        self.name=name

        # Optionally tie weights as in:
        # "Using the Output Embedding to Improve Language Models" (Press & Wolf 2016)
        # https://arxiv.org/abs/1608.05859
        # and
        # "Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling" (Inan et al. 2016)
        # https://arxiv.org/abs/1611.01462
        if tie_weights:
            if hidden_size != embed_dim:
                raise ValueError('When using the tied flag, nhid must be equal to emsize')
            self.fc.weight = self.emb.weight

    def init_hidden(self, bsize):
        return (torch.zeros(self.n_layers, bsize, self.hidden_size).to(self.device),
                torch.zeros(self.n_layers, bsize, self.hidden_size).to(self.device))

    def prompt(self, text: str, max_length=50, argm=False):
        self.eval()
        # encode, turn to tensor, and turn dimensions into batchable dimensions
        input_tensor = torch.tensor(self.tokenizer.encode(text), dtype=torch.long).unsqueeze(0).to(self.device)
        hidden = self.init_hidden(input_tensor.size(0))
        output = []

        with torch.no_grad():
            for _ in range(max_length):
                logits, hidden = self.forward(input_tensor, hidden)
                # dim = (batchsize, token seq len, vocab size)
                logits = logits[:,-1, :]

                #probs = torch.nn.functional.softmax(logits, dim=-1)
                #next_token_id = torch.multinomial(probs, 1).item()                
                if not argm: 
                    next_token_id = sampler(10, logits, top_p=0.8)
                else:
                    next_token_id = torch.argmax(logits, dim=-1)

                if self.tokenizer.eos_id() == next_token_id:
                    print('early stopping')
                    break

                output.append(next_token_id.item())

                input_tensor = torch.tensor([[next_token_id]], dtype=torch.long).to(self.device)

                
        return self.tokenizer.decode(output, out_type=str)


    def forward(self, x, hidden=None):
        x = self.emb(x)
        lstm_output, hidden = self.lstm(x, hidden)
        out = self.fc(lstm_output)
        return out, hidden

    def reps(self, train_kit):
        optimizer = train_kit['opt']
        training_loader = train_kit['train_loader']
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        epochs = train_kit['epochs']
        patience_counter = 0  # Track epochs without improvement
        patience=5
        best_vloss = 1_000_000.

        training_loss, val_loss = [], []
        hidden = self.init_hidden(self.batch_size)
        for epoch in tqdm(range(epochs)):
            # Make sure gradient tracking is on, and do a pass over the data
            self.train()
            running_loss = 0.
            for inputs, labels in training_loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                # Zero your gradients for every batch!
                optimizer.zero_grad()
                hidden = tuple([each.data for each in hidden])
                logits, hidden = self.forward(inputs, hidden)
                # logits shape should be (batch_size, seqlen, vocabsize)
                # labels should be (batch_size, expected_seq)
                #print(logits.shape)
                #print(labels.shape)
                loss = self.criterion(logits.view(-1, self.output_size), labels.view(-1))
                #print("loss backward")
                loss.backward()
                #print("optimizer")
                torch.nn.utils.clip_grad_norm_(self.parameters(), 1)
                # Adjust learning weights
                optimizer.step()

                # Gather data and report
                running_loss += loss.item()
                #print(f"EPOCH {epoch+1} Loss: {loss.item()}")

            avg_train_loss = running_loss / len(training_loader)
            training_loss.append(avg_train_loss)

            # Set the model to evaluation mode, disabling dropout and using population
            # statistics for batch normalization.
            self.eval()
            running_vloss = 0.0

            vhidden = self.init_hidden(self.batch_size)
            # Disable gradient computation and reduce memory consumption.
            with torch.no_grad():
                for vinputs, vlabels  in train_kit['val_loader']: 
                    vinputs, vlabels = vinputs.to(self.device), vlabels.to(self.device)
                    vhidden = tuple([each.data for each in vhidden])
                    voutputs, _ = self.forward(vinputs, vhidden)
                    vloss = self.criterion(voutputs.view(-1, self.output_size), vlabels.view(-1))
                    running_vloss += vloss.item()
            avg_vloss = running_vloss / len(train_kit['val_loader'])
            val_loss.append(avg_vloss)
            print('numbers = (avg loss {} avgval {} bestval {})'.format(avg_train_loss, avg_vloss, best_vloss))

            # Track best performance, and save the model's state
            if avg_vloss < best_vloss:
                print('patience reset')
                best_vloss = avg_vloss
                patience_counter=0
            else:
                print('patience counted increased')
                patience_counter+=1

            if patience_counter >= patience:
                print('not patience anymore. early stopping')
                break

            train_kit['scheduler'].step(avg_vloss)

        model_path = 'model_{}_{}.torch'.format(timestamp, self.name)
        torch.save(self.state_dict(), model_path)
        with open('./model_{}_{}_{}.json'.format(timestamp, self.name, 'tvl'), 'w+') as f:
            json.dump([training_loss,val_loss], f)
        
        return training_loss, val_loss


