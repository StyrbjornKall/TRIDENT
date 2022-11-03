import torch
import torch.nn as nn

import copy
import pickle as pkl
import random
from typing import List


def GPUinfo(device):
    print(f"GPUs on node: {torch.cuda.get_device_name()}")
    print(f"Number of GPUs available: {torch.cuda.device_count()}")
    print(f"Using {device} device")
    t = torch.cuda.get_device_properties(device).total_memory
    r = torch.cuda.memory_reserved(device)
    a = torch.cuda.memory_allocated(device)
    f = t-r-a
    print(f'{np.round(f/1000000000, 2)} Gb free on CUDA')


class Modify_architecture:
    """
    Class to modify different parts of the Pytorch model. 
    Used to 
    - Apply Layerwise Learning Rate Decay (LLRD)
    - Freeze RoBERTa encoders and embedding
    - Reinitialize pre-trained RoBERTa encoder weights and biases 
    """
    def __init__(self, model):
        self.initializer_range = model.roberta.config.initializer_range

    def LLRD(self, model, init_lr) -> List:
    
        opt_parameters = []    # To be passed to the optimizer (only parameters of the layers you want to update).
        named_parameters = list(model.named_parameters()) 
            
        # According to AAAMLP book by A. Thakur, we generally do not use any decay 
        # for bias and LayerNorm.weight layers.
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        lr = init_lr
                    
        # === 6 Hidden layers ==========================================================
        
        for layer in range(5,-1,-1):        
            params_0 = [p for n,p in named_parameters if f"encoder.layer.{layer}." in n 
                        and any(nd in n for nd in no_decay)]
            params_1 = [p for n,p in named_parameters if f"encoder.layer.{layer}." in n 
                        and not any(nd in n for nd in no_decay)]
            
            layer_params = {"params": params_0, "lr": lr, "weight_decay": 0.0}
            opt_parameters.append(layer_params)   
                                
            layer_params = {"params": params_1, "lr": lr, "weight_decay": 0.01}
            opt_parameters.append(layer_params)       
            
            lr *= 0.9     
            
        # === Embedding layer ==========================================================
        
        params_0 = [p for n,p in named_parameters if "embeddings" in n 
                    and any(nd in n for nd in no_decay)]
        params_1 = [p for n,p in named_parameters if "embeddings" in n
                    and not any(nd in n for nd in no_decay)]
        
        embed_params = {"params": params_0, "lr": lr, "weight_decay": 0.0} 
        opt_parameters.append(embed_params)
            
        embed_params = {"params": params_1, "lr": lr, "weight_decay": 0.01} 
        opt_parameters.append(embed_params)     

        # === Linear layer ==========================================================
        params_1 = [p for n,p in named_parameters if "fc" in n]
        linear_params = {"params": params_1, "lr": init_lr, "weight_decay": 0.0}
        opt_parameters.append(linear_params)
        
        return opt_parameters

    def FreezeModel(self, model, freeze_layer_count: int = 0, freeze_embedding: bool = False):
        if freeze_layer_count != -1:
            # if freeze_layer_count == -1, we only freeze the embedding layer
            # otherwise we freeze the first `freeze_layer_count` encoder layers
            for layer in model.roberta.encoder.layer[:freeze_layer_count]:
                for param in layer.parameters():
                    param.requires_grad = False

        if freeze_embedding:
            for param in model.roberta.embeddings.parameters():
                param.requires_grad = False

        return model


    def ReinitializeEncoderLayers(self, model, reinit_n_layers: int=0):
        """
        Reinitializes the weights and biases in the **last** n encoder layers in RoBERTa.
        I.e. reinit_n_layers=1 will reinitialize the last encoder.
        """
        # Re-init last n layers.
        for n in range(reinit_n_layers):
            model.roberta.encoder.layer[-(n+1)].apply(self._init_weight_and_bias)

        return model
            
    def _init_weight_and_bias(self, module):                        
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)  


class DNN_module(nn.Module):
    def __init__(self, one_hot_enc_len, n_hidden_layers, layer_sizes, dropout, activation):
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
        
        x = x.squeeze()
        return x

class fishbAIT(nn.Module):
    '''
    Class to build Transformer and DNN structure. Supports maximum 3 hidden layers and RoBERTa transformers.
    '''
    def __init__(self, roberta, dnn):
        super(fishbAIT, self).__init__()
        self.roberta = roberta 
        self.dnn = dnn
        
    def forward(self, sent_id, mask, exposure_duration, one_hot_encoding):
        roberta_output = self.roberta(sent_id, attention_mask=mask)[0]#.detach()#[:,0,:]#.detach() # all samples in batch : only CLS embedding : entire embedding dimension

        roberta_output = roberta_output[:,0,:]
      
        inputs = torch.cat((roberta_output, torch.t(exposure_duration.unsqueeze(0)), one_hot_encoding), 1)
        out = self.dnn(inputs)
        
        return out
