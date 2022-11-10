import torch
from transformers import AutoModel, AutoTokenizer
import pandas as pd
import numpy as np
from inference_utils.model_utils import DNN_module, fishbAIT_model
from inference_utils.pytorch_data_utils import PreProcessDataForInference, BuildInferenceDataLoaderAndDataset
from tqdm import tqdm
from typing import List, TypeVar
PandasDataFrame = TypeVar('pandas.core.frame.DataFrame')
PyTorchDataLoader = TypeVar('torch.utils.data.DataLoader')


class fishbAIT:
    def __init__(self, model_version: str='EC50EC10', path_to_model_weights=None, device=None):
        self.model_version = model_version
        self.path_to_model_weights = path_to_model_weights

        if device != None:
            self.device = device
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        if self.model_version == 'EC50':
            self.list_of_effects = ['MOR']
            self.list_of_endpoints = ['EC50']
        elif self.model_version == 'EC10':
            self.list_of_effects = ['MOR','DVP','ITX','REP','MPH','POP','GRO']
            self.list_of_endpoints = ['EC10']
        elif self.model_version == 'EC50EC10':
            self.list_of_effects = ['MOR','DVP','ITX','REP','MPH','POP','GRO']
            self.list_of_endpoints = ['EC50','EC10']

    def load_fine_tuned_model(self):
    
        self.roberta = AutoModel.from_pretrained(f'StyrbjornKall/fishbAIT_{self.model_version}', use_auth_token='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')
        self.tokenizer = AutoTokenizer.from_pretrained(f'StyrbjornKall/fishbAIT_{self.model_version}', use_auth_token='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')

        if self.model_version == 'EC50':
            dnn = DNN_module(one_hot_enc_len=1, n_hidden_layers=3, layer_sizes=[700,500,300])
        elif self.model_version == 'EC10':
            dnn = DNN_module(one_hot_enc_len=7, n_hidden_layers=3, layer_sizes=[700,500,300])
        elif self.model_version == 'EC50EC10':
            dnn = DNN_module(one_hot_enc_len=9, n_hidden_layers=3, layer_sizes=[700,500,300])
        
        self.dnn = self.__loadcheckpoint__(dnn, self.model_version, self.path_to_model_weights)

        self.fishbAIT_model = fishbAIT_model(self.roberta, self.dnn)

    def __loadcheckpoint__(self, dnn, version, path):
        try:
            if path != None:
                checkpoint_dnn = torch.load(f'{path}final_model_{version}_dnn_saved_weights.pt', map_location=self.device)
            else:
                checkpoint_dnn = torch.load(f'../fishbAIT/final_model_{version}_dnn_saved_weights.pt', map_location=self.device)
        except:
            raise FileNotFoundError(
                f'''Tried to load DNN module from path 
                ../fishbAIT/final_model_{version}_dnn_saved_weights.pt
                but could not find file. Please specify the full path to the saved model.''')

        dnn.load_state_dict(checkpoint_dnn)
        
        return dnn


    def predict_toxicity(self, SMILES, exposure_duration: int, endpoint: str, effect: str):
        if isinstance(SMILES, pd.DataFrame):
            if 'smiles' in SMILES.columns:
                SMILES.rename(columns={'smiles': 'SMILES'}, inplace=True)
            if 'Smiles' in SMILES.columns:
                SMILES.rename(columns={'Smiles': 'SMILES'}, inplace=True)
        elif isinstance(SMILES, list):
            SMILES = pd.DataFrame(SMILES, columns=['SMILES'])

        SMILES['exposure_duration'] = exposure_duration
        SMILES['endpoint'] = endpoint
        SMILES['effect'] = effect

        processor = PreProcessDataForInference(dataframe=SMILES)
        processor.GetCanonicalSMILES()
        processor.GetOneHotEnc(list_of_endpoints=self.list_of_endpoints, list_of_effects=self.list_of_effects)
        processed_data = processor.dataframe

        loader = BuildInferenceDataLoaderAndDataset(
            processed_data, 
            variables=['SMILES_Canonical_RDKit', 'exposure_duration', 'OneHotEnc_concatenated'], 
            tokenizer = self.tokenizer).dataloader

        self.fishbAIT_model.eval()
        preds = []
        for _, batch in enumerate(tqdm(loader)):
            with torch.no_grad():
                preds.append(self.fishbAIT_model(*batch.values()).numpy())

        preds = np.concatenate(preds, axis=0)
        SMILES['predictions log10(mg/L)'] = preds
        SMILES['predictions (mg/L)'] = 10**preds

        return SMILES
    