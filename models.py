from torch import nn
import torch
from datetime import datetime
from torch.autograd import Variable
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
        hidden = Variable(torch.zeros(self.rnn.num_layers, batch_size, self.hidden_size))
        # initialize hidden state with zero weights, and move to GPU if available
        return hidden

    def reps(self, train_kit):
        loss_fn = train_kit['loss']
        optimizer = train_kit['opt']
        training_loader = train_kit['train_loader']
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        EPOCHS = train_kit['epochs']
        best_vloss = 1_000_000.

        training_loss, val_loss = [], []
        hidden = self.init_hidden(self.batch_size)
        for epoch in range(EPOCHS):
            # Make sure gradient tracking is on, and do a pass over the data
            self.train(True)
            running_loss = 0.
            for inputs, labels in training_loader:
                # Zero your gradients for every batch!
                optimizer.zero_grad()
                # print(labels[0])
                hidden = tuple([each.data for each in hidden])
                logits, hidden = self.forward(inputs, hidden)
                logits = torch.sum(logits, dim=1)
                print(logits.shape, labels.shape)
                loss = loss_fn(logits.view(-1, self.output_size), labels.view(-1))
                #print("loss backward")
                loss.backward()
                #print("optimizer")
                torch.nn.utils.clip_grad_norm_(self.parameters(), 3)
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
            self.eval()
            # Disable gradient computation and reduce memory consumption.
            with torch.no_grad():
                for vinputs, vlabels  in train_kit['val_loader']: 
                    voutputs, _ = self.forward(vinputs)
                    vloss = loss_fn(voutputs.view(-1, self.output_size), vlabels.view(-1))
                    running_vloss += vloss

            avg_vloss = running_vloss / len(train_kit['val_loader'])
            val_loss.append(avg_vloss)
            print('averages = (loss {} val {})'.format(avg_train_loss, avg_vloss))

            # Track best performance, and save the model's state
            if avg_vloss < best_vloss:
                best_vloss = avg_vloss

            train_kit['scheduler'].step(best_vloss)

        model_path = 'model_{}_{}.torch'.format(timestamp, self.name)
        torch.save(self.state_dict(), model_path)
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
            pad_token_id=0,
            dropout=0.3):
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

    def init_hidden(self, bsize):
        return (torch.zeros(self.n_layers, bsize, self.hidden_size).to(self.device),
                torch.zeros(self.n_layers, bsize, self.hidden_size).to(self.device))

    def prompt(self, text: str, max_length=50):
        self.eval()
        # encode, turn to tensor, and turn dimensions into batchable dimensions
        input_tensor = torch.tensor(self.tokenizer.encode(text), dtype=torch.long).unsqueeze(0).to(self.device)
        hidden = self.init_hidden(1)
        output = []

        with torch.no_grad():
            for _ in range(max_length):
                logits, hidden = self.forward(input_tensor, hidden)
                # dim = (batchsize, token seq len, vocab size)
                logits = logits[:,-1, :]
                #probs = torch.nn.functional.softmax(logits, dim=-1)
                #next_token_id = torch.multinomial(probs, 1).item()                
                next_token_id = sampler(20, logits, top_p=0.8)
                output.append(next_token_id.item())

                input_tensor = torch.tensor([[next_token_id]], dtype=torch.long).to(self.device)

                
        return self.tokenizer.decode(output, out_type=str)


    def forward(self, x, hidden=None):
        x = self.emb(x)
        lstm_output, hidden = self.lstm(x, hidden)
        out = self.fc(lstm_output)
        return out, hidden

    def reps(self, train_kit):
        loss_fn = train_kit['loss']
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
                # print(labels[0])
                hidden = tuple([each.data for each in hidden])
                logits, hidden = self.forward(inputs, hidden)
                # logits shape should be (batch_size, seqlen, vocabsize)
                # labels should be (batch_size, expected_token_id)
                #print(logits.shape)
                #print(labels.shape)
                loss = loss_fn(logits.view(-1, self.output_size), labels.view(-1))
                #print("loss backward")
                loss.backward()
                #print("optimizer")
                torch.nn.utils.clip_grad_norm_(self.parameters(), 5)
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
                    vloss = loss_fn(voutputs.view(-1, self.output_size), vlabels.view(-1))
                    running_vloss += vloss
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
                model_path = 'model_{}_{}.torch'.format(timestamp, self.name)
                print('saving to', model_path)
                torch.save(self.state_dict(), model_path)
                break

            train_kit['scheduler'].step(best_vloss)

        model_path = 'model_{}_{}.torch'.format(timestamp, self.name)
        torch.save(self.state_dict(), model_path)
        return training_loss, val_loss


