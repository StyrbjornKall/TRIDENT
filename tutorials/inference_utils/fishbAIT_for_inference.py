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
        
        self.__check_allowed_prediction__(endpoint, effect)

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


    def __check_allowed_prediction__(self, endpoint, effect):

        if endpoint not in self.list_of_endpoints:
            raise RuntimeError(f'''You are trying to predict a `{endpoint}` endpoint with fishbAIT version {self.model_version}. 
            This will not work. Reload a correct version of fishbAIT (i.e. `EC50`, `EC10` or `EC50EC10`) or specify correct endpoint.
            For additional information call: __help__''')
        
        if effect not in self.list_of_effects:
            raise RuntimeError(f'''You are trying to predict a `{effect}` effect with fishbAIT version {self.model_version}. 
            This will not work. Reload a correct version of fishbAIT (i.e. `EC50`, `EC10` or `EC50EC10`) or specify correct effect.
            For additional information call: __help__''')

    
    def __help__(self):
        print('''
        This is a python class used to load and use the fine-tuned deep-learning model `fishbAIT` for environmental toxicity predictions in fish.
        The models have been trained on a large corpus of SMILES (chemical representations) on data collected from various sources.

        Currently there are three models available for use.
        - `EC50` The EC50 model is trained on EC50 mortality (MOR) data and is thus suitable for the prediction of said endpoints.
        - `EC10` The EC10 model is trained on EC10/NOEC data with various effects (mortality, intoxication, development, reproduction, morphology, growth and population) ab. (MOR, ITX, DVP, REP, MPH, GRO, POP)
        - `EC50EC10` The EC50EC10 model is trained on both EC50, EC10 and NOEC data with various effects (mortality, intoxication, development, reproduction, morphology, growth and population) ab. (MOR, ITX, DVP, REP, MPH, GRO, POP)

        For the most accurate predictions, refer to the combined EC50EC10 model.
        
        LOADING A MODEL:
        Load the model by initiating this class with the desired model version.

        MAKING PREDICTIONS:
        Making predictions is easy. Just  and use the `predict_toxicity` function to make a prediction on a list of SMILES.
        
        ''')