import torch
from transformers import AutoModel, AutoTokenizer
import pandas as pd
import numpy as np
from inference_utils.model_utils import DNN_module, TRIDENT
from inference_utils.pytorch_data_utils import PreProcessDataForInference, BuildInferenceDataLoaderAndDataset
from tqdm import tqdm
from typing import List, TypeVar
PandasDataFrame = TypeVar('pandas.core.frame.DataFrame')
PyTorchDataLoader = TypeVar('torch.utils.data.DataLoader')


class TRIDENT_for_inference:
    def __init__(self, model_version: str='EC50EC10_fish', path_to_model_weights=None, device=None):

        self.model_version = model_version
        self.path_to_model_weights = path_to_model_weights

        if device != None:
            self.device = device
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        effectordering = {
            'EC50_algae': ['POP'],
            'EC10_algae': ['POP'],
            'EC50EC10_algae': ['POP'], 
            'EC50_invertebrates': ['MOR','ITX'],
            'EC10_invertebrates': ['MOR','DVP','ITX','REP','MPH','POP'],
            'EC50EC10_invertebrates': ['MOR','DVP','ITX','REP','MPH','POP'],
            'EC50_fish': ['MOR'],
            'EC10_fish': ['MOR','DVP','ITX','REP','MPH','POP','GRO'],
            'EC50EC10_fish': ['MOR','DVP','ITX','REP','MPH','POP','GRO'] 
            }
        
        endpointordering = {
            'EC50_algae': ['EC50'],
            'EC10_algae': ['EC10'],
            'EC50EC10_algae': ['EC50','EC10'], 
            'EC50_invertebrates': ['EC50'],
            'EC10_invertebrates': ['EC10'],
            'EC50EC10_invertebrates': ['EC50','EC10'],
            'EC50_fish': ['EC50'],
            'EC10_fish': ['EC10'],
            'EC50EC10_fish': ['EC50','EC10'] 
            }

        self.list_of_effects = effectordering[self.model_version]
        self.list_of_endpoints = endpointordering[self.model_version]

        self.text_placeholder = st.empty()

    def load_fine_tuned_model(self):

        onehotencodinglengths = {
            'EC50_algae': 1,
            'EC10_algae': 1,
            'EC50EC10_algae': 2, 
            'EC50_invertebrates': 2,
            'EC10_invertebrates': 6,
            'EC50EC10_invertebrates': 8,
            'EC50_fish': 1,
            'EC10_fish': 7,
            'EC50EC10_fish': 9
        }

        self.roberta = AutoModel.from_pretrained(f'StyrbjornKall/{self.model_version}')
        self.tokenizer = AutoTokenizer.from_pretrained(f'StyrbjornKall/{self.model_version}')

        dnn = DNN_module(one_hot_enc_len=onehotencodinglengths[self.model_version], n_hidden_layers=3, layer_sizes=[700,500,300], dropout=0.2)
        
        self.dnn = self.__loadcheckpoint__(dnn, self.model_version, self.path_to_model_weights)

        self.TRIDENT_model = TRIDENT(self.roberta, self.dnn).to(self.device)

    def __loadcheckpoint__(self, dnn, version, path):
        print('Loading DNN... \n')
        try:
            if path != None:
                checkpoint_dnn = torch.load(f'{path}final_model_{version}_dnn_saved_weights.pt', map_location=self.device)
            else:
                path = f'./TRIDENT/final_model_{version}_dnn_saved_weights.pt'
                checkpoint_dnn = torch.load(f'{path}', map_location=self.device)
        except Exception as E:
            raise FileNotFoundError(
                f'''Tried to load DNN module from path 
                ../TRIDENT/final_model_{version}_dnn_saved_weights.pt
                but could not find file. Please specify the full path to the saved model.''')

        dnn.load_state_dict(checkpoint_dnn)
        
        return dnn

    def predict_toxicity(self, SMILES, exposure_duration: int, endpoint: str, effect: str, return_cls_embeddings: bool=False):
        
        self.__check_allowed_prediction__(endpoint, effect)

        if isinstance(SMILES, pd.DataFrame):
            if 'smiles' in SMILES.columns:
                SMILES.rename(columns={'smiles': 'SMILES'}, inplace=True)
            if 'Smiles' in SMILES.columns:
                SMILES.rename(columns={'Smiles': 'SMILES'}, inplace=True)
        elif isinstance(SMILES, list):
            SMILES = pd.DataFrame(SMILES, columns=['SMILES'])

        SMILES['exposure_duration log10(h)'] = np.log10(exposure_duration)
        SMILES['endpoint'] = endpoint
        SMILES['effect'] = effect

        processor = PreProcessDataForInference(dataframe=SMILES)
        processor.GetCanonicalSMILES()
        processor.GetOneHotEnc(list_of_endpoints=self.list_of_endpoints, list_of_effects=self.list_of_effects)
        processed_data = processor.dataframe

        loader = BuildInferenceDataLoaderAndDataset(
            processed_data, 
            variables=['SMILES_Canonical_RDKit', 'exposure_duration log10(h)', 'OneHotEnc_concatenated'], 
            tokenizer = self.tokenizer).dataloader

        self.TRIDENT_model.eval()
        preds = []
        cls_embeddings = []
        n_batches = len(loader)
        for _, batch in enumerate(tqdm(loader)):
            with torch.no_grad():
                pred, cls = self.TRIDENT_model(*batch.to(self.device).values())
                preds.append(pred.cpu().numpy().astype(np.float32))
                cls_embeddings.append(cls.cpu().numpy().astype(np.float32))
        preds = np.concatenate(preds, axis=0)
        cls_embeddings = np.concatenate(cls_embeddings, axis=0).tolist()
        SMILES['predictions log10(mg/L)'] = preds
        SMILES['predictions (mg/L)'] = 10**preds
        if return_cls_embeddings == True:
            SMILES['CLS_embeddings'] = cls_embeddings

        return SMILES


    def __check_allowed_prediction__(self, endpoint, effect):

        if endpoint not in self.list_of_endpoints:
            raise RuntimeError(f'''You are trying to predict a `{endpoint}` endpoint with TRIDENT version {self.model_version}. 
            This will not work. Reload a correct version of TRIDENT (i.e. `EC50`, `EC10` or `EC50EC10`) or specify correct endpoint.
            For additional information call: __help__''')
        
        if effect not in self.list_of_effects:
            raise RuntimeError(f'''You are trying to predict a `{effect}` effect with TRIDENT version {self.model_version}. 
            This will not work. Reload a correct version of TRIDENT (i.e. `EC50`, `EC10` or `EC50EC10`) or specify correct effect.
            For additional information call: __help__''')

    
    def __help__(self):
        print('''
        This is a python class used to load and use the fine-tuned deep-learning model `TRIDENT` for environmental toxicity predictions in fish algae and aquatic invertebrates.
        The models have been trained on a large corpus of SMILES (chemical representations) on data collected from various sources.

        Currently there are nine models available for use. The models are divided by toxicity endpoint and by species group. The models are the following:
        **Fish**
        - `EC50_fish` The EC50 model is trained on EC50 mortality (MOR) data and is thus suitable for the prediction of said endpoints.
        - `EC10_fish` The EC10 model is trained on EC10/NOEC data with various effects (mortality, intoxication, development, reproduction, morphology, growth and population) ab. (MOR, ITX, DVP, REP, MPH, GRO, POP)
        - `EC50EC10_fish` The EC50EC10 model is trained on EC50, EC10 and NOEC data with various effects (mortality, intoxication, development, reproduction, morphology, growth and population) ab. (MOR, ITX, DVP, REP, MPH, GRO, POP) and is the BEST model out of the three fish models.

        **(Aquatic) Invertebrates**
        - `EC50_invertebrates` The EC50 model is trained on EC50 mortality/intoxication (MOR, ITX) invertebrate data and is thus suitable for the prediction of said endpoints.
        - `EC10_invertebrates` The EC10 model is trained on EC10/NOEC data with various effects (mortality, intoxication, development, reproduction, morphology and population) ab. (MOR, ITX, DVP, REP, MPH, POP)
        - `EC50EC10_invertebrates` The EC50EC10 model is trained on EC50, EC10 and NOEC data with various effects (mortality, intoxication, development, reproduction, morphology and population) ab. (MOR, ITX, DVP, REP, MPH, POP) and is the BEST model out of the three invertebrate models.

        **Algae**
        - `EC50_algae` The EC50 model is trained on EC50 population (POP) algae data and is thus suitable for the prediction of said endpoints.
        - `EC10_algae` The EC10 model is trained on EC10/NOEC population (POP) algae data and is thus suitable for the prediction of said endpoints.
        - `EC50EC10_algae` The EC50EC10 model is trained on EC50, EC10 and NOEC population (POP) algae data and is the BEST out of the three algae models.

        For the most accurate predictions, always refer to the combined EC50EC10 model.
        
        LOADING A MODEL:
        Load the model by initiating this class with the desired model version, e.g. `EC50_algae`.

        MAKING PREDICTIONS:
        Making predictions is easy. Just  and use the `predict_toxicity` function to make a prediction on a list of SMILES.
        
        ''')
