import pandas as pd 
import numpy as np
from rdkit import Chem, RDLogger
RDLogger.DisableLog('rdApp.*')


def Preprocess10x10Fold(name, uselogdata: bool=True):
    concatenated_results = pd.read_pickle(f'../../data/results/{name}_predictions_100x_CV_RDKit.zip', compression='zip')
    
    RescaleDuration(concatenated_results)
    concatenated_results.rename(columns={'preds': 'fishbAIT'}, inplace=True)
    concatenated_results.loc[concatenated_results.endpoint == 'NOEC', 'endpoint'] = 'EC10'
    concatenated_results['residuals'] = concatenated_results.labels-concatenated_results.fishbAIT

    if uselogdata:
        pass
    else:
        for col in concatenated_results.columns:
            if col in ['mgperL','labels','fishbAIT','ECOSAR','TEST','VEGA']:
                try:
                    concatenated_results[col] = 10**concatenated_results[col]
                except:
                    pass

    avg_predictions = concatenated_results.groupby(['CAS', 'organism', 'Duration_Value',
        'effect', 'mgperL','endpoint', 'SMILES','SMILES_Canonical_RDKit',
        'cmpdname'], as_index=False, dropna=False).mean(numeric_only=True).drop(columns=['seed','fold_id','L1Error'])
    avg_predictions['prediction_std'] = concatenated_results.groupby(['CAS', 'organism', 'Duration_Value',
        'effect', 'mgperL','endpoint', 'SMILES','SMILES_Canonical_RDKit',
        'cmpdname'], as_index=False, dropna=False).std(numeric_only=True).fishbAIT
    avg_predictions['residuals'] = avg_predictions.labels-avg_predictions.fishbAIT
    avg_predictions['fishbAIT_residuals'] = avg_predictions['residuals']
    avg_predictions['CASRN'] = avg_predictions['CAS'].apply(lambda x: ''.join(x.split('-'))).astype(int)
    avg_predictions['Canonical_SMILES_figures'] = avg_predictions.SMILES_Canonical_RDKit.apply(lambda x: GetCanonicalSMILESForFigures(x))
    return avg_predictions

def GroupDataForPerformance(avg_predictions):

    medians = avg_predictions.groupby(['Duration_Value',
    'effect','endpoint', 'Canonical_SMILES_figures',
    'cmpdname'], as_index=False, dropna=False).median(numeric_only=True)
    counts = avg_predictions.groupby(['Duration_Value',
    'effect','endpoint', 'Canonical_SMILES_figures',
    'cmpdname'], as_index=False, dropna=False).count()
    
    counts.rename(columns={'labels': 'counts'}, inplace=True)

    medians['counts'] = counts['counts']
    for col in medians.columns:
        if col not in (['effect','endpoint', 'Canonical_SMILES_figures','cmpdname','counts']):
            medians[[col]] = medians[[col]]*medians[['counts']].to_numpy()

    mean = medians.groupby((['endpoint','Canonical_SMILES_figures','cmpdname']), as_index=False, dropna=False).sum(min_count=1, numeric_only=True)

    for col in mean.columns:
        try:
            if col not in ['endpoint','Canonical_SMILES_figures','cmpdname','counts']:
                mean[col] = mean[[col]]/mean[['counts']].to_numpy()
        except:
            pass
    mean['L1error'] = abs(mean['residuals'])
    return mean

def RescaleDuration(df):
    if isinstance(df.Duration_Value[0], float):
        df.Duration_Value = (10**df.Duration_Value).astype(int)

def DurationBinner(df, intervals):
    interval_match = dict(zip(intervals, ['short exposure', 'medium exposure', 'long exposure']))

    def find(x):
        for i in intervals:
            if x <= i:
                return interval_match[i]

    df['Duration_Value_binned'] = df.Duration_Value.apply(lambda x: find(x))

    return df


def GetCanonicalSMILESForFigures(smiles):
    try:
        return Chem.MolToSmiles(Chem.MolFromSmiles(smiles), canonical=True)
    except:
        return smiles