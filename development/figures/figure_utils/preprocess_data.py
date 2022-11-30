import pandas as pd 
import numpy as np

def Preprocess10x10Fold(name, uselogdata: bool=True):
    concatenated_results = pd.read_csv(f'../../data/results/{name}_predictions_100x_CV_RDKit.zip', compression='zip')
    RescaleDuration(concatenated_results)
    concatenated_results.rename(columns={'preds': 'fishbAIT'}, inplace=True)
    concatenated_results.endpoint[concatenated_results.endpoint == 'NOEC'] = 'EC10'
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
    
    if 'SMILES_Canonical_RDKit' in concatenated_results.columns: 
        avg_predictions = concatenated_results.groupby(['CAS', 'organism', 'Duration_Value',
            'effect', 'mgperL','endpoint', 'SMILES','SMILES_Canonical_RDKit',
            'cmpdname'], as_index=False, dropna=False).mean().drop(columns=['seed','fold_id','L1Error'])
        avg_predictions['prediction_std'] = concatenated_results.groupby(['CAS', 'organism', 'Duration_Value',
            'effect', 'mgperL','endpoint', 'SMILES','SMILES_Canonical_RDKit',
            'cmpdname'], as_index=False, dropna=False).std().fishbAIT
    else:
        avg_predictions = concatenated_results.groupby(['CAS', 'organism', 'Duration_Value',
            'effect', 'mgperL','endpoint', 'SMILES',
            'cmpdname'], as_index=False, dropna=False).mean().drop(columns=['seed','fold_id','L1Error'])
        avg_predictions['prediction_std'] = concatenated_results.groupby(['CAS', 'organism', 'Duration_Value',
            'effect', 'mgperL','endpoint', 'SMILES',
            'cmpdname'], as_index=False, dropna=False).std().fishbAIT
    avg_predictions['residuals'] = avg_predictions.labels-avg_predictions.fishbAIT
    return avg_predictions

def GroupDataForPerformance(avg_predictions):
    if 'SMILES_Canonical_RDKit' in avg_predictions.columns:
        medians = avg_predictions.groupby(['Duration_Value',
        'effect','endpoint', 'SMILES','SMILES_Canonical_RDKit',
        'cmpdname'], as_index=False, dropna=False).median()
        counts = avg_predictions.groupby(['Duration_Value',
        'effect','endpoint', 'SMILES','SMILES_Canonical_RDKit',
        'cmpdname'], as_index=False, dropna=False).count()
        
        counts.rename(columns={'labels': 'counts'}, inplace=True)

        medians['counts'] = counts['counts']
        for col in medians.columns:
            if col not in (['effect','endpoint', 'SMILES','SMILES_Canonical_RDKit','cmpdname','counts']):
                medians[[col]] = medians[[col]]*medians[['counts']].to_numpy()

        mean = medians.groupby((['endpoint','SMILES','SMILES_Canonical_RDKit','cmpdname']), as_index=False, dropna=False).sum(min_count=1)

        for col in mean.columns:
            try:
                if col not in ['endpoint','SMILES','SMILES_Canonical_RDKit','cmpdname','counts']:
                    mean[col] = mean[[col]]/mean[['counts']].to_numpy()
            except:
                pass
    else:
        medians = avg_predictions.groupby(['Duration_Value',
        'effect','endpoint', 'SMILES',
        'cmpdname'], as_index=False, dropna=False).median()
        counts = avg_predictions.groupby(['Duration_Value',
        'effect','endpoint', 'SMILES',
        'cmpdname'], as_index=False, dropna=False).count()

        counts.rename(columns={'labels': 'counts'}, inplace=True)

        medians['counts'] = counts['counts']
        for col in medians.columns:
            if col not in ['effect','endpoint', 'SMILES','cmpdname','counts']:
                medians[[col]] = medians[[col]]*medians[['counts']].to_numpy()

        mean = medians.groupby((['endpoint','SMILES','cmpdname']), as_index=False, dropna=False).sum(min_count=1)

        for col in mean.columns:
            try:
                if col not in ['endpoint','SMILES','cmpdname','counts']:
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