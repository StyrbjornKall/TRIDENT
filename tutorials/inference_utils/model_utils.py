import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer


class DNN_module(nn.Module):
    def __init__(self, one_hot_enc_len, n_hidden_layers, layer_sizes, dropout: float=0.2):
        super(DNN_module, self).__init__()
        self.one_hot_enc_len = one_hot_enc_len
        self.n_hidden_layers = n_hidden_layers
        self.layer_sizes = layer_sizes
        self.dropout = nn.Dropout(dropout)

        self.active =  nn.ReLU()
        self.fc1 = nn.Linear(768 + 1 + one_hot_enc_len, layer_sizes[0]) # This is where we have to add dimensions (+1) to fascilitate the additional parameters
        if n_hidden_layers == 1:
            self.fc2 = nn.Linear(layer_sizes[0],1)
        elif n_hidden_layers == 2:
            self.fc2 = nn.Linear(layer_sizes[0],  layer_sizes[1])
            self.fc3 = nn.Linear(layer_sizes[1], 1)
        elif n_hidden_layers == 3:
            self.fc2 = nn.Linear(layer_sizes[0],  layer_sizes[1])
            self.fc3 = nn.Linear(layer_sizes[1], layer_sizes[2])
            self.fc4 = nn.Linear(layer_sizes[2], 1)
        elif n_hidden_layers == 4:
            self.fc2 = nn.Linear(layer_sizes[0],  layer_sizes[1])
            self.fc3 = nn.Linear(layer_sizes[1], layer_sizes[2])
            self.fc4 = nn.Linear(layer_sizes[2], layer_sizes[3])
            self.fc5 = nn.Linear(layer_sizes[3], 1)

    def forward(self, inputs):
        if self.n_hidden_layers >= 1:
            x = self.fc1(inputs)
            x = self.active(x)
            x = self.dropout(x)
            x = self.fc2(x)
        if self.n_hidden_layers >= 2:
            x = self.active(x)
            x = self.dropout(x)
            x = self.fc3(x)
        if self.n_hidden_layers >= 3:
            x = self.active(x)
            x = self.dropout(x)
            x = self.fc4(x)
        if self.n_hidden_layers >= 4:
            x = self.active(x)
            x = self.dropout(x)
            x = self.fc5(x)
        
        x = x.squeeze(1)
        return x

class fishbAIT_model(nn.Module):
    '''
    Class to combine the Transformer (chemberta) and DNN module.
    '''
    def __init__(self, roberta=None, dnn=None):
        super(fishbAIT_model, self).__init__()
        self.roberta = roberta 
        self.dnn = dnn
        
    def forward(self, sent_id, mask, exposure_duration, one_hot_encoding):
        roberta_output = self.roberta(sent_id, attention_mask=mask)[0]#.detach()#[:,0,:]#.detach() # all samples in batch : only CLS embedding : entire embedding dimension

        roberta_output = roberta_output[:,0,:]
      
        inputs = torch.cat((roberta_output, torch.t(exposure_duration.unsqueeze(0)), one_hot_encoding), 1)
        out = self.dnn(inputs)
        
        return out