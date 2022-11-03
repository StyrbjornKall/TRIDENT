
from tqdm import tqdm, tqdm_pandas
import pandas as pd
pd.options.mode.chained_assignment = None
import numpy as np
import pickle as pkl

import requests
from itertools import chain
from typing import List
from rdkit import Chem, RDLogger
RDLogger.DisableLog('rdApp.*')


# Get data for model contains functions for data preprocessing in order to train a transformer model for ecotoxicity prediction.
# These utils functions are imported and used in the main script.

class PreprocessData():
    tqdm.pandas()

    def __init__(self, dataframe):
        self.dataframe = dataframe

    def FilterData(self, concentration_thresh: int, endpoint: List[str], effect: List[str], species_groups: List[str], concentration_sign: str=None, log_data: bool=True, drop_columns: bool=True):

        '''
        Function to filter out unwanted data from pandas dataframe. Note, requires specific column names to work.

        Columns names should be as follows: 
        COMBINED_endpoint: e.g. 'EC50'
        COMBINED_effect: e.g. 'MOR'
        COMBINED_mgperL: concentration at endpoint
        COMBINED_species_group: e.g. 'fish', 'insects', 'algae'
        COMBINED_Duration_Value: duration (h)
        COMBINED_Conc_sign: '=', '>', '<'
        
        '''

        self.dataframe.insert(0, 'internal_id', range(len(self.dataframe))) 
        self.dataframe = self.dataframe[self.dataframe.COMBINED_endpoint.isin(endpoint)] if 'all' not in endpoint else self.dataframe
        self.dataframe = self.dataframe[self.dataframe.COMBINED_effect.isin(effect)] if 'all' not in effect else self.dataframe
        self.dataframe = self.dataframe[self.dataframe.COMBINED_Duration_Value > 0]
        self.dataframe = self.dataframe[(self.dataframe.COMBINED_mgperL < concentration_thresh) & (self.dataframe.COMBINED_mgperL > 0)]
        self.dataframe = self.dataframe[self.dataframe.COMBINED_species_group.isin(species_groups)] if species_groups[0] != 'all' else self.dataframe
        self.dataframe.COMBINED_organism = self.dataframe.COMBINED_organism.apply(lambda x: self._GetSpecies(x))
        
        if log_data == True:
            self.dataframe.COMBINED_mgperL = np.log10(self.dataframe.COMBINED_mgperL)
            self.dataframe.COMBINED_Duration_Value = np.log10(self.dataframe.COMBINED_Duration_Value)

        if concentration_sign != None:
            self.dataframe = self.dataframe[self.dataframe.COMBINED_Conc_sign == concentration_sign]

        if drop_columns == True:
            for col in ['smudge_reference', 'COMBINED_DOI', 'COMBINED_Duration_Unit', 'COMBINED_pubmed_ID']:
                try:
                    self.dataframe = self.dataframe.drop(columns=[col])
                except:
                    pass

        self.dataframe = self.dataframe.reset_index().drop(columns='index', axis=1)
        return self.dataframe



    def GetOneHotSpeciesGroup(self, list_of_species_groups: List[str]):
        '''
        Builds one hot encoded numpy arrays for given list of species groups, i.e. fish/invertebrates/algae etc.

        Groups crustaceans and invertebrates by renaming crustaceans --> invertebrates.
        '''
        if 'crustaceans' in list_of_species_groups:
            print(f"Renamed crustaceans *invertebrates* in {sum(self.dataframe['COMBINED_species_group'] == 'crustaceans')} positions")
            self.dataframe.loc[self.dataframe.COMBINED_species_group == 'crustaceans', 'COMBINED_species_group'] = 'invertebrates'
            list_of_species_groups.remove('crustaceans')

        if len(list_of_species_groups) > 1:
            hot_enc_dict = dict(zip(list_of_species_groups, np.eye(len(list_of_species_groups)).astype(int).tolist()))
            self.dataframe = self.dataframe.reset_index().drop(columns='index', axis=1)
            try:
                clas = self.dataframe.COMBINED_species_group.apply(lambda x: self._Match(x, list_of_species_groups))
                encoded_clas = clas.apply(lambda x: np.array(hot_enc_dict[x]))
                self.dataframe['OneHotEnc_species_group'] = encoded_clas
            except:
                raise Exception('An unexpected error occurred.')

        else:
            print('''Did not return onehotencoding for Species groups. Why? You specified only one species group.''')

        return self.dataframe




    def GetOneHotEndpoint(self, endpoints: List[str]):
        '''
        Builds one hot encoded numpy arrays for given endpoints. Groups EC10 and NOEC measurements by renaming EC10 --> NOEC.
        '''
        list_of_endpoints = endpoints.copy()
        if 'EC10' in list_of_endpoints:
            print(f"Renamed EC10 *NOEC* in {sum(self.dataframe['COMBINED_endpoint'] == 'EC10')} positions")
            self.dataframe.loc[self.dataframe.COMBINED_endpoint == 'EC10', 'COMBINED_endpoint'] = 'NOEC'
            list_of_endpoints.remove('EC10')
            
        if len(list_of_endpoints) > 1:
            hot_enc_dict = dict(zip(list_of_endpoints, np.eye(len(list_of_endpoints), dtype=int).tolist()))
            self.dataframe = self.dataframe.reset_index().drop(columns='index', axis=1)
            try:
                clas = self.dataframe.COMBINED_endpoint.apply(lambda x: self._Match(x, list_of_endpoints))
                encoded_clas = clas.apply(lambda x: np.array(hot_enc_dict[x]))
                self.dataframe['OneHotEnc_endpoint'] = encoded_clas
            except:
                raise Exception('An unexpected error occurred.')

        else:
            print('''Did not return onehotencoding for Endpoint. Why? You specified only one Endpoint or you specified NOEC and EC10 which are coded to be the same endpoint.''')

        return self.dataframe



    def GetOneHotEffect(self, list_of_effects: List[str]):
        '''
        Builds one hot encoded numpy arrays for given effects.
        '''

        if len(list_of_effects) > 1:
            hot_enc_dict = dict(zip(list_of_effects, np.eye(len(list_of_effects), dtype=int).tolist()))
            self.dataframe = self.dataframe.reset_index().drop(columns='index', axis=1)
            try:
                clas = self.dataframe.COMBINED_effect.apply(lambda x: self._Match(x, list_of_effects))
                encoded_clas = clas.apply(lambda x: np.array(hot_enc_dict[x]))
                self.dataframe['OneHotEnc_effect'] = encoded_clas
            except:
                raise Exception('An unexpected error occurred.')

        else:
            print('''Did not return onehotencoding for Effect. Why? You specified only one Effect.''')

        return self.dataframe




    def GetLineages(self, path_dict_of_species_and_lineage):
        '''
        Retrieves lineages produced by ncbi taxonomy by querying a locally stored dictionary using species name inside the dataset. 
        '''
        self.dataframe = self.dataframe.reset_index().drop(columns='index', axis=1)
        try:
            file = open(path_dict_of_species_and_lineage, 'rb')
            dict_of_species_and_lineage = pkl.load(file)
            self.dataframe['Lineage'] = self.dataframe.COMBINED_organism.apply(lambda x: dict_of_species_and_lineage[x][0] if x in dict_of_species_and_lineage.keys() else [])
            file.close()
        except OSError as e:
            print(f'Could not load dictionary. {e}') 

        return self.dataframe




    def GetOneHotSpeciesClass(self, groups: List[str]):
        '''
        Builds one hot encoded numpy arrays for given species groups. Requires dataframe to contain Lineage column.

        Automatically assigns species not in list as other
        '''
        
        if len(groups) > 1:
            try:
                groups.append('other')

                hot_enc_dict = dict(zip(groups, np.eye(len(groups)).astype(int).tolist()))
                self.dataframe = self.dataframe.reset_index().drop(columns='index', axis=1)
                clas = self.dataframe.Lineage.apply(lambda x: self._Match(x, groups))
                encoded_clas = clas.apply(lambda x: np.array(hot_enc_dict[x]))
                n_other = np.sum(clas=='other')
                
                print(f'Classified {n_other} as other')
                self.dataframe['Species_class'] = clas
                self.dataframe['OneHotEnc_Species_class'] = encoded_clas
            except:
                raise Exception('Did not get any classes or there are no lineages present in the dataframe.')
        else:
            print('''Did not return onehotencoding for Species classes. Why? No specified classes.''')

        return self.dataframe



    def ConcatenateOneHotEnc(self):
        '''
        Concatenates all one hot encodings into one numpy vector. Preferably use this as input to network.
        '''
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




    def GetPubchemCID(self, path_dict_of_SMILES_and_CID: str, drop_missing_entries: bool=True):
        '''
        Retrieves PubChem CIDs by querying a locally stored dictionary using SMILES inside the dataset. 
        
        If this does not succeed, raises error and asks for permission to download from pubchem.
        '''
        try:
            dict_of_SMILES = pkl.load(open(path_dict_of_SMILES_and_CID, 'rb'))
        
        except OSError:
            print('Could not load dictionary.') 
            commence_download = input('''Do you wish to retrieve CIDs from PubChem? [y/n]''')

            if commence_download in ['y', 'Yes', 'YES', 'yes']:
                unique_SMILES = self.dataframe.SMILES.unique()
                dict_of_SMILES = dict.fromkeys(unique_SMILES)

                for smile in tqdm(unique_SMILES):
                    dict_of_SMILES[smile] = self._GetCID(smile)

                self.dataframe['Pubchem_CID'] = self.dataframe.SMILES.apply(lambda x: dict_of_SMILES[x])
                pkl.dump(dict_of_SMILES, open(path_dict_of_SMILES_and_CID, 'wb'))

        self.dataframe = self.dataframe.reset_index().drop(columns='index', axis=1)
        CID_list = np.zeros(len(self.dataframe)).astype(int)
        dropable = []
        for idx, entry in self.dataframe.SMILES.iteritems():
            if entry in dict_of_SMILES.keys():
                CID_list[idx] = dict_of_SMILES[entry]
            else:
                CID_list[idx] = 0 # 0 corresponds to invalid CID
                dropable.append(idx)

        if drop_missing_entries == True:  
            self.dataframe = self.dataframe.drop(dropable)
            CID_list = CID_list[CID_list != 0]
            print(f'Dropped {len(dropable)} entries from dataframe due to SMILES not having CID')
            self.dataframe['Pubchem_CID'] = CID_list.tolist()            
        else:
            self.dataframe['Pubchem_CID'] = CID_list.tolist()

        self.dataframe = self.dataframe.reset_index().drop(columns='index', axis=1)

        return self.dataframe
        


    def GetPubchemSMILES(self, path_CID_metadata: str):
        '''
        Retrieves Canonical SMILES using PubChem CIDs. Subfunction of GetMetadata.
        '''
        self.dataframe = self.GetMetadata(path_CID_metadata, list_of_metadata=['isosmiles'])
        self.dataframe = self.dataframe.rename(columns={'isosmiles': 'SMILES_Canonical_pubchem'})

        return self.dataframe



    def GetCanonicalSMILES(self):
        '''
        Retrieves Canonical SMILES using RDKit.
        '''
        self.dataframe['SMILES_Canonical_RDKit'] = self.dataframe.SMILES.apply(lambda x: self.__CanonicalizeRDKit(x))

        return self.dataframe



    def GetMetadata(self, path_CID_metadata: str, list_of_metadata: List[str]=['mw', 'polararea','complexity','xlogp','hbonddonor','hbondacc','isosmiles', 'cmpdname']):
        '''
        Retrieves additional metadata by pubchem ID.
        '''
        metadata = pd.read_csv(path_CID_metadata)
        merged = pd.merge(self.dataframe, metadata, left_on='Pubchem_CID', right_on='cid')

        merged = merged[self.dataframe.columns.tolist()+list_of_metadata] if list_of_metadata[0] != 'all' else merged
        self.dataframe = merged
        self.dataframe = self.dataframe.rename(columns={'isosmiles': 'Canonical_SMILES'})
        
        return self.dataframe




    ## Convenience functions
    def _Match(self, x, groups):
        try:
            clas = [y for y in groups if y in x][0]
        except:
            clas = 'other'
        return clas

    def _GetCID(self, smiles):
        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/{smiles}/cids/TXT"
        try:
            r = requests.get(url)
            r.raise_for_status()
            cid = int(r.text[:-1])
        except:
            cid = None
        return cid

    def _GetSpecies(self, x):
        try:
            x = ' '.join(x.split()[0:2])
        except:
            x = 'unspecified'
        return x

    def _GetMolFromSMILES(self, smiles):
        try:
            return wandb.Molecule.from_smiles(smiles)
        except:
            return None

    def __CanonicalizeRDKit(self, smiles):
        try:
            return Chem.MolToSmiles(Chem.MolFromSmiles(smiles), canonical=True)
        except:
            return smiles