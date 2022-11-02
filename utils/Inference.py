import pandas as pd
import pickle as pkl
pd.options.mode.chained_assignment = None
import numpy as np
from typing import List, TypeVar
from rdkit import Chem, RDLogger

import torch
from torch.utils.data import Dataset, DataLoader, SequentialSampler

from transformers import DataCollatorWithPadding

RDLogger.DisableLog('rdApp.*')
PandasDataFrame = TypeVar('pandas.core.frame.DataFrame')
PyTorchDataLoader = TypeVar('torch.utils.data.DataLoader')


class PreProcessDataForInference():
    '''
    PreProcessDataForInference contains functions to:
    - generate one hot encoding vectors to represent endpoints and effects such that the main model class can make accurate predictions
    - Canonicalize SMILES to the RDKit format used during both pre-training and fine-tuning 

    If the data consists only of a single endpoint or effect no one hot encoding will be generated.
    '''
    def __init__(self, dataframe):
        self.dataframe = dataframe

    def GetOneHotEnc(self, list_of_endpoints: List[str], list_of_effects: List[str]):
        '''
        Concatenates all one hot encodings into one numpy vector. Preferably use this as input to network.
        '''
        self.dataframe = self._GetOneHotEndpoint(list_of_endpoints=list_of_endpoints)
        self.dataframe = self._GetOneHotEffect(list_of_effects=list_of_effects)

        try:
            columns = self.dataframe.filter(like='OneHotEnc').columns.tolist()
            temp1 = np.array([el.tolist() for el in self.dataframe[columns[0]].values])
            for idx, col in enumerate(columns):
                try:
                    temp2 = np.array([el.tolist() for el in self.dataframe[columns[idx+1]].values])
                    temp1 = np.concatenate((temp1,temp2), axis=1)
                except:
                    pass
            self.dataframe['OneHotEnc_concatenated'] = temp1.tolist()
        except:
            self.dataframe['OneHotEnc_concatenated'] = np.zeros((len(self.dataframe), 1)).tolist()
            print('''Will use input 0 to network due to no Onehotencodings being present.''')

        return self.dataframe

    
    def GetCanonicalSMILES(self):
        '''
        Retrieves Canonical SMILES using RDKit.
        '''
        self.dataframe['SMILES_Canonical_RDKit'] = self.dataframe.SMILES.apply(lambda x: self.__CanonicalizeRDKit(x))

        return self.dataframe

    
    def _GetOneHotEndpoint(self, list_of_endpoints: List[str]):
        '''
        Builds one hot encoded numpy arrays for given endpoints. Groups EC10 and NOEC measurements by renaming EC10 --> NOEC.
        '''
        if 'EC10' in list_of_endpoints:
            print(f"Renamed EC10 *NOEC* in {sum(self.dataframe['endpoint'] == 'EC10')} positions")
            self.dataframe.loc[self.dataframe.endpoint == 'EC10', 'endpoint'] = 'NOEC'
            list_of_endpoints.remove('EC10')
    
        if len(list_of_endpoints) > 1:
            encoding_order = ['EC50', 'NOEC']
            hot_enc_dict = dict(zip(encoding_order, np.eye(len(encoding_order), dtype=int).tolist()))
            self.dataframe = self.dataframe.reset_index().drop(columns='index', axis=1)
            try:
                clas = self.dataframe.Endpoint.apply(lambda x: self._Match(x, list_of_endpoints))
                encoded_clas = clas.apply(lambda x: np.array(hot_enc_dict[x]))
                self.dataframe['OneHotEnc_endpoint'] = encoded_clas
            except:
                raise Exception('An unexpected error occurred.')

        else:
            print('''Did not return onehotencoding for Endpoint. Why? You specified only one Endpoint or you specified NOEC and EC10 which are coded to be the same endpoint.''')

        return self.dataframe

    def _GetOneHotEffect(self, list_of_effects: List[str]):
        '''
        Builds one hot encoded numpy arrays for given effects.
        '''
        if len(list_of_effects) > 1:
            effect_order = ['MOR','DVP','ITX','REP','MPH','POP','GRO']
            hot_enc_dict = dict(zip(effect_order, np.eye(len(effect_order), dtype=int).tolist()))
            self.dataframe = self.dataframe.reset_index().drop(columns='index', axis=1)
            try:
                clas = self.dataframe.effect.apply(lambda x: self._Match(x, list_of_effects))
                encoded_clas = clas.apply(lambda x: np.array(hot_enc_dict[x]))
                self.dataframe['effect'] = encoded_clas
            except:
                raise Exception('An unexpected error occurred.')

        else:
            print('''Did not return onehotencoding for Effect. Why? You specified only one Effect.''')

        return self.dataframe

    ## Convenience functions
    def _Match(self, x, groups):
        try:
            clas = [y for y in groups if y in x][0]
        except:
            clas = 'other'
        return clas

    def __CanonicalizeRDKit(self, smiles):
        try:
            return Chem.MolToSmiles(Chem.MolFromSmiles(smiles), canonical=True)
        except:
            return smiles


class Inference_dataset(Dataset):
    '''
    Class for efficient loading of data for inference.

    *Variables should be a list with column names corresponding to the column containing
    1. SMILES
    2. Exposure Duration
    3. Endpoint
    4. Effect

    In the specified order.
    '''
    def __init__(self, df: PandasDataFrame, variables: List[str], tokenizer):
        self.df = df
        self.tokenizer = tokenizer
        self.variables = variables

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        encodings = self.tokenizer.encode_plus(row[self.variables[0]], truncation=True, max_length=self.max_len)
        ids = torch.tensor(encodings['input_ids'])
        mask = torch.tensor(encodings['attention_mask'])
        dur = torch.tensor(row[self.variables[1]], dtype=torch.float32)
        onehot = torch.tensor(row[self.variables[2]], dtype=torch.float32)
        sample = {'input_ids': enc['input_ids'], 'attention_mask': enc['attention_mask'], 'duration': dur, 'onehotenc': onehot}

        return sample

class BuildInferenceDataLoader():
    '''
    Class to build PyTorch Dataloader for batch inference. 
    The Dataloader uses a SequentialSampler
    '''
    def __init__(self, df: PandasDataFrame, variables: List[str], batch_size: int, max_length: int=100, seed: int, test_size: float, tokenizer):
        self.df = df
        self.bs = batch_size
        self.tokenizer = tokenizer
        self.max_len = max_length
        self.variables = variables
        self.label = label
        self.seed = seed
        self.collator = DataCollatorWithPadding(tokenizer=self.tokenizer, padding='longest', return_tensors='pt')
        
    def BuildDataset(self, df: PandasDataFrame) -> PandasDataFrame:
        dataset = Inference_dataset(df, self.tokenizer)
        return dataset

    def BuildValidationLoader(self, num_workers: int=0) -> PyTorchDataLoader:
        dataset = self.BuildDataset(self.df)
        sampler = SequentialSampler(dataset)
        dataloader = DataLoader(dataset, sampler=sampler, batch_size=self.bs, collate_fn=self.collator, num_workers=num_workers)

        return dataloader


class DNN_module(nn.Module):
    def __init__(self, one_hot_enc_len, n_hidden_layers, layer_sizes):
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