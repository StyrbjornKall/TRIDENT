import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from typing import List, TypeVar

class DNN_module(nn.Module):
    """
    # DNN module
    nn.Module class to generate DNN module. Can be used separately or as an addition to other models, e.g. as a transformer regression head.
    
    The network uses a ReLU activation function.

    ## Inputs
    - one_hot_enc_len: The length of the one-hot-encoding vector (set to 0 if not applicable)
    - layer_sizes: a list of layer sizes with same length as n_hidden_layers
    - n_hidden_layers: the number of hidden layers
    - dropout: the dropout rate of each hidden layer

    """
    def __init__(self, one_hot_enc_len: int, n_hidden_layers: int, layer_sizes: List[int], dropout: float):
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

class TRIDENT(nn.Module):
    '''
    # TRIDENT
    Class to build Transformer and DNN structure. Supports maximum 4 hidden layers and RoBERTa transformers.

    ## Inputs
    - roberta: a pytorch transformer (BERT-like) model
    - dnn: an instance of the DNN_module class
    '''
    def __init__(self, roberta, dnn):
        super(TRIDENT, self).__init__()
        self.roberta = roberta 
        self.dnn = dnn
        
    def forward(self, sent_id, mask, exposure_duration, one_hot_encoding):
        roberta_output = self.roberta(sent_id, attention_mask=mask)[0] # Last hidden state NOT pooler output

        roberta_output = roberta_output[:,0,:] #all samples in batch : only CLS embedding : entire embedding dimension
      
        inputs = torch.cat((roberta_output, torch.t(exposure_duration.unsqueeze(0)), one_hot_encoding), 1)
        out = self.dnn(inputs)
        
        return out, roberta_output
