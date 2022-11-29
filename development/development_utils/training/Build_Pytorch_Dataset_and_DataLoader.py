import tqdm as tqdm
import pickle as pkl
import pandas as pd
import numpy as np
import random
import requests
from itertools import chain
from typing import List, TypeVar

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, SequentialSampler, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

from collections import Counter

from transformers import DataCollatorWithPadding

from rdkit import Chem, RDLogger
RDLogger.DisableLog('rdApp.*')

PandasDataFrame = TypeVar('pandas.core.frame.DataFrame')
PyTorchDataLoader = TypeVar('torch.utils.data.DataLoader')


class Make_KFolds:
    """
    Builds a pandas dataframe with unique SMILES as index and K columns, one per fold, containing bools for whether the SMILES is used for training in the fold or not.
    """
    def __init__(self):
        pass
    def Split(self, smiles: PandasDataFrame, k_folds: int, seed: int) -> PandasDataFrame:
        kfold = KFold(n_splits = k_folds, shuffle = True, random_state = seed)
        
        unique_smiles = smiles.unique()
        blank_fold = np.zeros(len(unique_smiles))

        folds = pd.DataFrame(data=np.transpose(np.array([unique_smiles])), columns=['SMILES'])

        for fold in range(1,k_folds+1):
            folds[f'fold_{fold}'] = blank_fold

        for d, (train, test) in enumerate(kfold.split(unique_smiles)):
            folds[f'fold_{d+1}'].iloc[train] = True
            folds[f'fold_{d+1}'].iloc[test] = False
        
        return folds


class BuildDataLoader_KFold:
    '''
    Class to build PyTorch Dataloader for training and validation sets. Does not support test set.
    Builds Dataloader based on splitting all unique SMILES into K-folds.
    The Dataloader supports:
    - SequentialSampler
    - WeightedRandomSampler (both with weights as 1/n and 1/sqrt(n))
    '''
    def __init__(self, df: PandasDataFrame, folds: PandasDataFrame, fold_id: int, wandb_config: dict, label: str, batch_size: int, max_length: int, seed: int, tokenizer):
        self.df = df
        self.bs = batch_size
        self.tokenizer = tokenizer
        self.max_len = max_length
        self.config = wandb_config
        self.variables = self.config.inputs
        self.label = label
        self.seed = seed
        self.folds = folds
        if self.tokenizer != None:
            self.collator = DataCollatorWithPadding(tokenizer=self.tokenizer, padding='longest', return_tensors='pt')
        else:
            self.collator = None
        
        self.train = df[df[self.config.smiles_col_name].isin(folds.SMILES[folds[f'fold_{fold_id}']==True])]
        self.val = df[df[self.config.smiles_col_name].isin(folds.SMILES[folds[f'fold_{fold_id}']==False])]
        
    def BuildDataset(self, df: PandasDataFrame):
        if self.tokenizer != None:
            dataset = SMILES_dataset(df, self.variables, self.label, self.tokenizer, self.max_len)
        else:
            dataset = CLS_dataset(df, self.variables, self.label)
        return dataset


    def BuildTrainingLoader(self,  sampler_choice: str='WRS', num_workers: int=0, weight_args: List[str]=None) -> PyTorchDataLoader:
        '''
        weight_args: list defined in order [SMILES_col_name, effect_col_name, endpoint_col_name]
        '''
        if weight_args == None:
            # Builds only based on SMILES
            counts = Counter(self.train[self.config.smiles_col_name])
            weights = self.train[self.config.smiles_col_name].apply(lambda x: 1/counts[x]).tolist()
        else:
            counts = Counter(list(zip(self.train[weight_args[0]].tolist(), self.train[weight_args[1]].tolist(), self.train[weight_args[2]].tolist())))
            weights = 1/np.array([counts[i] for i in list(zip(self.train[weight_args[0]].tolist(), self.train[weight_args[1]].tolist(), self.train[weight_args[2]].tolist()))])

        if sampler_choice == 'WRS_sqrt':
            weights = np.sqrt(weights)

        samples_weight = torch.from_numpy(np.array(weights))
        
        dataset = self.BuildDataset(self.train)
        sampler = WeightedRandomSampler(samples_weight, len(samples_weight), replacement = True)
        train_dataloader = DataLoader(dataset, sampler=sampler, batch_size=self.bs, num_workers=num_workers, collate_fn=self.collator)

        print(f'Built training dataloader with {len(dataset)} samples')
        return train_dataloader

    def BuildValidationLoader(self, sampler_choice: str, num_workers: int=0) -> PyTorchDataLoader:
        dataset = self.BuildDataset(self.val)
        if sampler_choice == 'WeightedRandomSampler':
            counts = Counter(self.val[self.config.smiles_col_name])
            weights = self.val[self.config.smiles_col_name].apply(lambda x: 1/counts[x]).tolist()
            samples_weight = sum(counts.values())*torch.from_numpy(np.array(weights))
            sampler = WeightedRandomSampler(samples_weight, len(samples_weight), replacement = True)
        else:
            sampler = SequentialSampler(dataset)

        val_dataloader = DataLoader(dataset, sampler=sampler, batch_size=self.bs, num_workers=num_workers, collate_fn=self.collator)

        print(f'Built validation dataloader with {len(dataset)} samples')
        return val_dataloader

    def MakeVenn(self, A, B) -> set:
        AB_overlap = A & B
        A_rest = A - AB_overlap
        B_rest = B - AB_overlap
        AB_only = AB_overlap

        sets = Counter()               #set order A, B, C   
        sets['10'] = len(A_rest)      #100 denotes A on, B off, C off 
        sets['01'] = len(B_rest)      #010 denotes A off, B on, C off
        sets['11'] = len(AB_only)     #110 denotes A on, B on, C off

        return sets


class BuildDataLoader_with_trainval_ratio:
    '''
    Class to build PyTorch Dataloader for training and validation sets. Does not support test set.
    Builds Dataloader based on splitting all unique SMILES with train/val ratio.
    The Dataloader supports:
    - SequentialSampler
    - WeightedRandomSampler (both with weights as 1/n and 1/sqrt(n))
    '''
    def __init__(self, df: PandasDataFrame, wandb_config, label: str, batch_size: int, max_length: int, seed: int, test_size: float, tokenizer):
        self.df = df
        self.bs = batch_size
        self.tokenizer = tokenizer
        self.max_len = max_length
        self.config = wandb_config
        self.variables = self.config.inputs
        self.label = label
        self.seed = seed
        if self.tokenizer != None:
            self.collator = DataCollatorWithPadding(tokenizer=self.tokenizer, padding='longest', return_tensors='pt')
        else:
            self.collator = None

        if test_size != 0:
            self.train, self.val = train_test_split(df.SMILES.unique().tolist(), test_size=test_size, random_state=seed)
            self.train = df[df.SMILES.isin(self.train)]
            self.val = df[df.SMILES.isin(self.val)]
        else:
            self.train = self.df
        
    def BuildDataset(self, df: PandasDataFrame):
        if self.tokenizer != None:
            dataset = SMILES_dataset(df, self.variables, self.label, self.tokenizer, self.max_len)
        else:
            dataset = CLS_dataset(df, self.variables, self.label)
        return dataset

    def BuildTrainingLoader(self,  sampler_choice: str='WRS', num_workers: int=0, weight_args: List[str]=None) -> PyTorchDataLoader:
        '''
        weight_args: list defined in order [SMILES_col_name, effect_col_name, endpoint_col_name]
        '''
        if weight_args == None:
            # Builds only based on SMILES
            counts = Counter(self.train[self.config.smiles_col_name])
            weights = self.train[self.config.smiles_col_name].apply(lambda x: 1/counts[x]).tolist()
        else:
            counts = Counter(list(zip(self.train[weight_args[0]].tolist(), self.train[weight_args[1]].tolist(), self.train[weight_args[2]].tolist())))
            weights = 1/np.array([counts[i] for i in list(zip(self.train[weight_args[0]].tolist(), self.train[weight_args[1]].tolist(), self.train[weight_args[2]].tolist()))])

        if sampler_choice == 'WRS_sqrt':
            weights = np.sqrt(weights)

        samples_weight = torch.from_numpy(np.array(weights))
        
        dataset = self.BuildDataset(self.train)
        sampler = WeightedRandomSampler(samples_weight, len(samples_weight), replacement = True)
        train_dataloader = DataLoader(dataset, sampler=sampler, batch_size=self.bs, num_workers=num_workers, collate_fn=self.collator)

        print(f'Built training dataloader with {len(dataset)} samples')
        return train_dataloader

    def BuildValidationLoader(self, sampler_choice: str, num_workers: int=0) -> PyTorchDataLoader:
        dataset = self.BuildDataset(self.val)
        if sampler_choice == 'WeightedRandomSampler':
            counts = Counter(self.val[self.config.smiles_col_name])
            weights = self.val[self.config.smiles_col_name].apply(lambda x: 1/counts[x]).tolist()
            samples_weight = sum(counts.values())*torch.from_numpy(np.array(weights))
            sampler = WeightedRandomSampler(samples_weight, len(samples_weight), replacement = True)
        else:
            sampler = SequentialSampler(dataset)

        val_dataloader = DataLoader(dataset, sampler=sampler, batch_size=self.bs, num_workers=num_workers, collate_fn=self.collator)

        print(f'Built validation dataloader with {len(dataset)} samples')
        return val_dataloader

    def MakeVenn(self, A, B) -> set:
        AB_overlap = A & B
        A_rest = A - AB_overlap
        B_rest = B - AB_overlap
        AB_only = AB_overlap

        sets = Counter()               #set order A, B, C   
        sets['10'] = len(A_rest)      #100 denotes A on, B off, C off 
        sets['01'] = len(B_rest)      #010 denotes A off, B on, C off
        sets['11'] = len(AB_only)     #110 denotes A on, B on, C off

        return sets


class SMILES_dataset(Dataset):
    '''
    Class for efficient loading of data

    Expects variables to be defined in following order:
    1. SMILES
    2. Duration
    3. Onehotencoding
    '''
    def __init__(self, df: PandasDataFrame, variables: List[str], label: str, tokenizer, max_len):
        self.df = df
        self.tokenizer = tokenizer
        self.variables = variables
        self.label = label
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        encodings = self.tokenizer.encode_plus(row[self.variables[0]], truncation=True, max_length=self.max_len)
        ids = torch.tensor(encodings['input_ids'])
        mask = torch.tensor(encodings['attention_mask'])
        dur = torch.tensor(row[self.variables[1]], dtype=torch.float32)
        onehot = torch.tensor(row[self.variables[2]], dtype=torch.float32)
        labels = torch.tensor(row[self.label], dtype=torch.float32)
        sample = {'input_ids': ids, 'attention_mask': mask, 'duration': dur, 'onehotenc': onehot, 'labels': labels}

        return sample

class CLS_dataset(Dataset):
    '''
    Class for efficient loading of data

    Expects variables to be defined in following order:
    1. CLS_embeddings
    2. Duration
    3. Onehotencoding
    '''
    def __init__(self, df: PandasDataFrame, variables: List[str], label: str):
        self.df = df
        self.variables = variables
        self.label = label

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        cls_embedding = torch.tensor(row[self.variables[0]], dtype=torch.float32)
        dur = torch.tensor(row[self.variables[1]], dtype=torch.float32)
        onehot = torch.tensor(row[self.variables[2]], dtype=torch.float32)
        labels = torch.tensor(row[self.label], dtype=torch.float32)
        sample = {'cls_embedding': cls_embedding, 'duration': dur, 'onehotenc': onehot, 'labels': labels}

        return sample

