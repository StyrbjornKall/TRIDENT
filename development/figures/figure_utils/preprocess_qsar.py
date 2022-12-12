from .figure_flags import *
from .preprocess_data import DurationBinner, GetCanonicalSMILESForFigures
import pandas as pd 
import numpy as np

# MAIN FUNCTIONS
## LOAD QSAR TOOL DATASETS
def LoadQSAR(endpoint, species_group):    
    if endpoint == 'EC50':
        VEGA = pd.read_pickle('../../data/results/VEGA_EC50_fish_invertebrates_algae.zip', compression='zip')
        ECOSAR = pd.read_pickle('../../data/results/ECOSAR_EC50_min_fish_invertebrates_algae.zip', compression='zip')
        TEST = pd.read_pickle('../../data/results/TEST_fish_invertebrates.zip', compression='zip')
    elif endpoint == 'EC10':
        VEGA = pd.read_pickle('../../data/results/VEGA_NOEC_fish_invertebrates_algae.zip', compression='zip')
        ECOSAR = pd.read_pickle('../../data/results/ECOSAR_ChV_min_fish_invertebrates_algae.zip', compression='zip')
        ECOSAR['Concentration (mg/L)'] = ECOSAR['Concentration (mg/L)']/np.sqrt(2)
        TEST = VEGA.copy()[0:0]
    elif endpoint == 'EC50EC10':
        VEGA_ec10 = pd.read_pickle('../../data/results/VEGA_NOEC_fish_invertebrates_algae.zip', compression='zip')
        ECOSAR_ec10 = pd.read_pickle('../../data/results/ECOSAR_ChV_min_fish_invertebrates_algae.zip', compression='zip')
        ECOSAR_ec10['Concentration (mg/L)'] = ECOSAR_ec10['Concentration (mg/L)']/np.sqrt(2)

        VEGA_ec50 = pd.read_pickle('../../data/results/VEGA_EC50_fish_invertebrates_algae.zip', compression='zip')
        ECOSAR_ec50 = pd.read_pickle('../../data/results/ECOSAR_EC50_min_fish_invertebrates_algae.zip', compression='zip')
        TEST = pd.read_pickle('../../data/results/TEST.zip', compression='zip')
        ECOSAR = pd.concat([ECOSAR_ec10,ECOSAR_ec50], axis=0)
        VEGA = pd.concat([VEGA_ec10,VEGA_ec50], axis=0)
        del [ECOSAR_ec10, ECOSAR_ec50, VEGA_ec10, VEGA_ec50]

    if species_group == 'fish':
        ECOSAR = ECOSAR[ECOSAR.Organism=='Fish']
        TEST = TEST[TEST.model_organism=='Fish']
        VEGA = VEGA[VEGA.model_organism.isin(['Fish','Fathead Minnow','Guppy'])]
    if species_group == 'invertebrates':
        ECOSAR = ECOSAR[ECOSAR.Organism=='Daphnid']
        TEST = TEST[TEST.model_organism=='Daphnia_magna']
        VEGA = VEGA[VEGA.model_organism.isin(['Daphnia Magna'])]
    if species_group == 'algae':
        ECOSAR = ECOSAR[ECOSAR.Organism=='Green Algae']
        VEGA = VEGA[VEGA.model_organism.isin(['Algae'])]
        TEST = VEGA.copy()[0:0]

    ECOSAR['Canonical_SMILES_figures'] = ECOSAR.original_SMILES.apply(lambda x: GetCanonicalSMILESForFigures(x))
    VEGA['Canonical_SMILES_figures'] = VEGA.original_SMILES.apply(lambda x: GetCanonicalSMILESForFigures(x))
    TEST['Canonical_SMILES_figures'] = TEST.original_SMILES.apply(lambda x: GetCanonicalSMILESForFigures(x))

    return ECOSAR, VEGA, TEST   

## CONVENIENCE FUNCTION FOR PREPROCESSING ALL QSAR DATA
def PrepareQSARData(ECOSAR, TEST, VEGA, inside_AD, remove_experimental):
    ECOSAR_tmp = ECOSAR.copy()
    TEST_tmp = TEST.copy()
    VEGA_tmp = VEGA.copy()
    ECOSAR_tmp.rename(columns={'Concentration (mg/L)': 'value'}, inplace=True)
    VEGA_tmp.value = np.log10(VEGA_tmp.value)
    ECOSAR_tmp.value = np.log10(ECOSAR_tmp.value)

    VEGA_tmp.replace([np.inf, -np.inf], np.nan, inplace=True)
    ECOSAR_tmp.replace([np.inf, -np.inf], np.nan, inplace=True)

    VEGA_tmp = VEGA_tmp[~VEGA_tmp.value.isna()]
    ECOSAR_tmp = ECOSAR_tmp[~ECOSAR_tmp.value.isna()]

    if not TEST_tmp.empty:
        TEST_tmp.value = np.log10(TEST_tmp.value)
        TEST_tmp.replace([np.inf, -np.inf], np.nan, inplace=True)
        TEST_tmp = TEST_tmp[~TEST_tmp.value.isna()]
        
    if inside_AD:
        ECOSAR_tmp = RemoveOutOfAD(ECOSAR_tmp, QSAR_type='ECOSAR')
        VEGA_tmp = RemoveOutOfAD(VEGA_tmp, QSAR_type='VEGA')
        try:
            TEST_tmp = RemoveOutOfAD(TEST_tmp, QSAR_type='TEST')
        except:
            pass

    if remove_experimental:
        VEGA_tmp = RemoveExperimentalData(VEGA_tmp, QSAR_type='VEGA')
        ECOSAR_tmp = RemoveExperimentalData(ECOSAR_tmp, QSAR_type='ECOSAR')
        try:
            TEST_tmp = RemoveExperimentalData(TEST_tmp, QSAR_type='TEST')
        except:
            pass

    VEGA_tmp = BestQSARPrediction(VEGA_tmp, QSAR_type='VEGA')
    ECOSAR_tmp = BestQSARPrediction(ECOSAR_tmp, QSAR_type='ECOSAR')
    try:
        TEST_tmp = BestQSARPrediction(TEST_tmp, QSAR_type='TEST')
    except:
        pass

    return ECOSAR_tmp, TEST_tmp, VEGA_tmp

## MATCH QSAR PREDICTIONS TO SMILES IN OUR DATASET
def MatchQSAR(df, ECOSAR_dict, TEST_dict, VEGA_dict, endpoint: str, species_group: str, duration: str=None):

    def Match(x, QSAR_dict):
        try:
            return QSAR_dict[x]
        except:
            return None
        
    if endpoint == 'EC50':
        if species_group == 'fish':
            df = df[df.Duration_Value == 96]
        if species_group == 'invertebrates':
            df = df[df.Duration_Value == 48]
        if species_group == 'algae':
            df = df[(df.Duration_Value == 96) | (df.Duration_Value == 71)]

    elif endpoint in ['EC10', 'NOEC', 'ChV']:
        df = DurationBinner(df, [170, 680, np.Inf])
        df = df[df.Duration_Value_binned.isin(duration)]

    df['ECOSAR'] = df.Canonical_SMILES_figures.apply(lambda x: Match(x, ECOSAR_dict))
    df['TEST'] = df.Canonical_SMILES_figures.apply(lambda x: Match(x, TEST_dict))
    df['VEGA'] = df.Canonical_SMILES_figures.apply(lambda x: Match(x, VEGA_dict))

    if species_group == 'algae':
        df['ECOSAR'][df['Duration_Value']==71] = None #Ecosar does not give outputs for 72h
        df['VEGA'][df['Duration_Value']==96] = None #Vega does not give outputs for 96h

    return df


# HELPER FUNCTIONS
## GET APPLICABILITY DOMAIN
def RemoveOutOfAD(df, QSAR_type: str):
    if QSAR_type == 'ECOSAR':
        tmp = len(df)
        # Remove CAS that are not profilable (from ECOSAR CAS prediction) and Alerts
        should_not_profile = pd.read_pickle('../../data/results/ECOSAR_v2.0_should_not_be_profiled_CAS_fish_invertebrates_algae.zip', compression='zip')
        should_not_profile['Canonical_SMILES_figures'] = should_not_profile.SMILES.apply(lambda x: GetCanonicalSMILESForFigures(x))
        df = df[(~df.CAS.isin(should_not_profile.CAS) & (df.Alert.isin([' '])) | df.Alert.isin(['  AcuteToChronicRatios', '  SaturateSolublity']))]
        # Remove SMILES that are not profilable (from ECOSAR CAS prediction)
        df = df[~df.Canonical_SMILES_figures.isin(should_not_profile.Canonical_SMILES_figures)]
        # Remove CAS that are not profilable (from ECOSAR SMILES prediction)
        should_not_profile = pd.read_pickle('../../data/results/ECOSAR_v2.0_should_not_be_profiled_SMILES_fish_invertebrates_algae.zip', compression='zip')
        should_not_profile['Canonical_SMILES_figures'] = should_not_profile.SMILES.apply(lambda x: GetCanonicalSMILESForFigures(x))
        df = df[~df.CAS.isin(should_not_profile.CAS)]
        # Remove SMILES that are not profilable (from ECOSAR SMILES prediction)
        df = df[~df.Canonical_SMILES_figures.isin(should_not_profile.Canonical_SMILES_figures)]
        print(f'Removed {tmp-len(df)} outside AD rows')
        return df
    elif QSAR_type == 'TEST':
        print(f'Removed 0 outside AD rows')
        return df
    elif QSAR_type == 'VEGA':
        tmp = len(df)
        df = df[df.reliability != 'low']
        print(f'Removed {tmp-len(df)} outside AD rows')
        return df

## REMOVE EXPERIMENTAL VALUES (TRAINING SET)
def RemoveExperimentalData(df, QSAR_type: str):
    if QSAR_type == 'ECOSAR':
        tmp = len(df)
        experimental_ecosar = pd.read_csv('../../data/results/ECOSAR_v2.2_CAS_experimental.zip', compression='zip')
        df = df[~df.CAS.isin(experimental_ecosar.ECOSAR_experimental_CAS.tolist())]
        print(f'Removed {tmp-len(df)} experimental rows')
    elif QSAR_type == 'TEST':
        tmp = len(df)
        experimental_smiles = df[df.reliability == 'EXPERIMENTAL'].Canonical_SMILES_figures.unique().tolist()
        df = df[~df.Canonical_SMILES_figures.isin(experimental_smiles)]
        print(f'Removed {tmp-len(df)} experimental rows')
    elif QSAR_type == 'VEGA':
        tmp = len(df)
        df = df[df.reliability != 'EXPERIMENTAL']
        print(f'Removed {tmp-len(df)} experimental rows')

    return df

## IF SEVERAL PREDICTIONS PER SMILES - GET THE BEST PREDICTIONS FOR EACH TOOL
def BestQSARPrediction(df, QSAR_type: str):
    if QSAR_type == 'ECOSAR':
        df = df.groupby(['Canonical_SMILES_figures'], as_index=False, dropna=False).min(numeric_only=True)
    elif QSAR_type == 'TEST':
        df = df.groupby(['Canonical_SMILES_figures'], as_index = False, dropna=False).min(numeric_only=True)
    elif QSAR_type == 'VEGA':
        remaining = df.copy()
        chosenvega = pd.DataFrame()
        for rely in ['EXPERIMENTAL','good','moderate','low']:
            chosenvega = pd.concat([remaining[remaining.reliability==rely], chosenvega])
            remaining = remaining[~remaining.Canonical_SMILES_figures.isin(chosenvega.Canonical_SMILES_figures)]
        df = chosenvega.groupby(['Canonical_SMILES_figures'], as_index=False).min(numeric_only=True)

    return df

