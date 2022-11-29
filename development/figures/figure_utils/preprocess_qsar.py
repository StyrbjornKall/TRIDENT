from .figure_flags import *
from .preprocess_data import DurationBinner
import pandas as pd 
import numpy as np

# MAIN FUNCTIONS
## LOAD QSAR TOOL DATASETS
def LoadQSAR(endpoint):    
    if endpoint == 'EC50':
        VEGA = pd.read_csv('C:/Users/skall/OneDrive - Chalmers/Documents/Ecotoxformer/Data/test_vega/VEGA_fish_EC50.csv')
        ECOSAR = pd.read_excel('C:/Users/skall/OneDrive - Chalmers/Documents/Ecotoxformer/Data/ecosar/ECOSAR_SMILES_fish_LD50_min.xlsx')
        TEST = pd.read_csv('C:/Users/skall/OneDrive - Chalmers/Documents/Ecotoxformer/Data/test_vega/TEST.csv')
        ECOSAR = ECOSAR[(ECOSAR.Duration == '96h')]
    elif endpoint == 'EC10':
        VEGA = pd.read_csv('C:/Users/skall/OneDrive - Chalmers/Documents/Ecotoxformer/Data/test_vega/VEGA_fish_NOEC.csv')
        ECOSAR = pd.read_excel('C:/Users/skall/OneDrive - Chalmers/Documents/Ecotoxformer/Data/ecosar/ECOSAR_SMILES_fish_ChV_min.xlsx')
        ECOSAR['Concentration (mg/L)'] = ECOSAR['Concentration (mg/L)']/np.sqrt(2)
        TEST = VEGA.copy()[0:0]
    elif endpoint == 'EC50EC10':
        VEGA_ec10 = pd.read_csv('C:/Users/skall/OneDrive - Chalmers/Documents/Ecotoxformer/Data/test_vega/VEGA_fish_NOEC.csv')
        ECOSAR_ec10 = pd.read_excel('C:/Users/skall/OneDrive - Chalmers/Documents/Ecotoxformer/Data/ecosar/ECOSAR_SMILES_fish_ChV_min.xlsx')
        ECOSAR_ec10['Concentration (mg/L)'] = ECOSAR_ec10['Concentration (mg/L)']/np.sqrt(2)
        VEGA_ec50 = pd.read_csv('C:/Users/skall/OneDrive - Chalmers/Documents/Ecotoxformer/Data/test_vega/VEGA_fish_EC50.csv')
        ECOSAR_ec50 = pd.read_excel('C:/Users/skall/OneDrive - Chalmers/Documents/Ecotoxformer/Data/ecosar/ECOSAR_SMILES_fish_LD50_min.xlsx')
        ECOSAR_ec50 = ECOSAR_ec50[(ECOSAR_ec50.Duration == '96h')]
        TEST = pd.read_csv('C:/Users/skall/OneDrive - Chalmers/Documents/Ecotoxformer/Data/test_vega/TEST.csv')
        ECOSAR = pd.concat([ECOSAR_ec10,ECOSAR_ec50], axis=0)
        VEGA = pd.concat([VEGA_ec10,VEGA_ec50], axis=0)
        del [ECOSAR_ec10, ECOSAR_ec50, VEGA_ec10, VEGA_ec50]

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
def MatchQSAR(df, ECOSAR_dict, TEST_dict, VEGA_dict, endpoint: str, duration: str=None):

    def Match(x, QSAR_dict):
        try:
            return QSAR_dict[x]
        except:
            return None
        
    try:
        df['CAS'] = df.COMBINED_CAS.apply(lambda x: x.replace('-','')).astype(int)
    except:
        pass

    if endpoint == 'EC50':
        df = df[df.COMBINED_Duration_Value == 96]
    elif endpoint in ['EC10', 'NOEC', 'ChV']:
        df = DurationBinner(df, [170, 680, np.Inf])
        df = df[df.Duration_Value_binned.isin(duration)]

    df['ECOSAR'] = df.SMILES.apply(lambda x: Match(x, ECOSAR_dict))
    df['TEST'] = df.SMILES.apply(lambda x: Match(x, TEST_dict))
    df['VEGA'] = df.SMILES.apply(lambda x: Match(x, VEGA_dict))

    return df


# HELPER FUNCTIONS
## GET APPLICABILITY DOMAIN
def RemoveOutOfAD(df, QSAR_type: str):
    if QSAR_type == 'ECOSAR':
        tmp = len(df)
        should_not_profile = pd.read_excel('C:/Users/skall/OneDrive - Chalmers/Documents/Ecotoxformer/Data/ecosar/should_not_be_profiled.xlsx')
        df = df[(~df.CAS.isin(should_not_profile.CAS) & (df.Alert.isin([' '])) | df.Alert.isin(['  AcuteToChronicRatios', '  SaturateSolublity']))]
        print(f'Removed {tmp-len(df)} rows')
        return df
    elif QSAR_type == 'TEST':
        print(f'Removed 0 rows')
        return df
    elif QSAR_type == 'VEGA':
        tmp = len(df)
        df = df[df.reliability != 'low']
        print(f'Removed {tmp-len(df)} rows')
        return df

## REMOVE EXPERIMENTAL VALUES (TRAINING SET)
def RemoveExperimentalData(df, QSAR_type: str):
    if QSAR_type == 'ECOSAR':
        experimental_ecosar = pd.read_csv('C:/Users/skall/OneDrive - Chalmers/Documents/Ecotoxformer/Data/ecosar/Experimental_CAS_ECOSAR_v2.2.csv')
        df = df[~df.CAS.isin(experimental_ecosar.ECOSAR_experimental_CAS.tolist())]
    elif QSAR_type == 'TEST':
        experimental_smiles = df[df.reliability == 'EXPERIMENTAL'].original_SMILES.unique().tolist()
        df = df[~df.original_SMILES.isin(experimental_smiles)]
    elif QSAR_type == 'VEGA':
        df = df[df.reliability != 'EXPERIMENTAL']

    return df

## IF SEVERAL PREDICTIONS PER SMILES - GET THE BEST PREDICTIONS FOR EACH TOOL
def BestQSARPrediction(df, QSAR_type: str):
    if QSAR_type == 'ECOSAR':
        df = df.groupby(['original_SMILES'], as_index=False).min()
    elif QSAR_type == 'TEST':
        df = df.groupby(['original_SMILES'], as_index = False).min()
    elif QSAR_type == 'VEGA':
        remaining = df.copy()
        chosenvega = pd.DataFrame()
        for rely in ['EXPERIMENTAL','good','moderate','low']:
            chosenvega = pd.concat([remaining[remaining.reliability==rely], chosenvega])
            remaining = remaining[~remaining.original_SMILES.isin(chosenvega.original_SMILES)]
        df = chosenvega.groupby(['original_SMILES'], as_index=False).min()

    return df