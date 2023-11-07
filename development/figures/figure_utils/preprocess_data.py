import pandas as pd 
import numpy as np
from rdkit import Chem, RDLogger
RDLogger.DisableLog('rdApp.*')
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances 
from sklearn.neighbors import NearestNeighbors

def Preprocess10x10Fold(name, uselogdata: bool=True, full_filepath = None, get_cosine_similarity = False):
    if full_filepath == None:
        concatenated_results = pd.read_pickle(f'../../data/results/{name}_predictions_100x_CV_RDKit.pkl.zip', compression='zip')
    else:
        concatenated_results = pd.read_pickle(full_filepath, compression='zip')

    if get_cosine_similarity == True:
        concatenated_results = GetCosineSimilarities(concatenated_results, name)
        
    RescaleDuration(concatenated_results)
    concatenated_results.rename(columns={'preds': 'TRIDENT'}, inplace=True)
    concatenated_results.loc[concatenated_results.endpoint == 'NOEC', 'endpoint'] = 'EC10'
    concatenated_results['residuals'] = concatenated_results.labels-concatenated_results.TRIDENT

    if uselogdata:
        pass
    else:
        for col in concatenated_results.columns:
            if col in ['mgperL','labels','TRIDENT','ECOSAR','TEST','VEGA']:
                try:
                    concatenated_results[col] = 10**concatenated_results[col]
                except:
                    pass

    avg_predictions = concatenated_results.groupby(['CAS', 'organism', 'Duration_Value',
        'effect', 'mgperL','endpoint', 'SMILES','SMILES_Canonical_RDKit',
        'cmpdname'], as_index=False, dropna=False).mean(numeric_only=True).drop(columns=['seed','fold_id','L1Error'])
    avg_predictions['prediction_std'] = concatenated_results.groupby(['CAS', 'organism', 'Duration_Value',
        'effect', 'mgperL','endpoint', 'SMILES','SMILES_Canonical_RDKit',
        'cmpdname'], as_index=False, dropna=False).std(numeric_only=True).TRIDENT
    avg_predictions['residuals'] = avg_predictions.labels-avg_predictions.TRIDENT
    avg_predictions['TRIDENT_residuals'] = avg_predictions['residuals']
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
    
def GetCosineSimilarities(concatenated_results, name):

    CLS_df = pd.read_pickle(f'../../data/results/{name}_final_model_training_data_RDkit.zip', compression='zip')
    CLS_df = CLS_df.drop_duplicates(subset=['SMILES_Canonical_RDKit'])[['SMILES_Canonical_RDKit', 'CLS_embeddings']]
    CLS_dict = dict(zip(CLS_df.SMILES_Canonical_RDKit, CLS_df.CLS_embeddings))

    concatenated_results['CLS_embedding_final_model'] = concatenated_results.SMILES_Canonical_RDKit.apply(lambda x: CLS_dict[x])

    cos_sim_max = []
    cos_sim_avg = []
    cos_sim_median = []
    cos_sim_knn = []
    concatenated_results.sort_values(by=['fold_id', 'seed'], inplace=True)
    for fold_id in range(1,11,1):
        for seed in range(41,51,1):
            fold = concatenated_results[(concatenated_results.fold_id==fold_id) & (concatenated_results.seed==seed)]
            train_smiles = GetTrainingSetSMILES(concatenated_results=concatenated_results, fold_id=fold_id, seed=seed)
            train_cls = list(map(lambda x: CLS_dict[x], train_smiles))
            
            max_cossim, avg_cossim, median_cossim, knn_cossim = CalculateCosineSimilarity(cls_training=train_cls, cls_validation=fold.CLS_embedding_final_model.tolist())

            cos_sim_max += max_cossim
            cos_sim_avg += avg_cossim
            cos_sim_median += median_cossim
            cos_sim_knn += knn_cossim

    concatenated_results['Cosine_sim_max'] = cos_sim_max
    concatenated_results['Cosine_sim_avg'] = cos_sim_avg
    concatenated_results['Cosine_sim_median'] = cos_sim_median
    concatenated_results['Cosine_sim_knn'] = cos_sim_knn
            
    return concatenated_results

def GetTrainingSetSMILES(concatenated_results, fold_id, seed):
    all_smiles = concatenated_results.drop_duplicates(subset=['SMILES_Canonical_RDKit'])
    validation_set_smiles = concatenated_results[(concatenated_results['fold_id']==fold_id) & (concatenated_results['seed']==seed)].SMILES_Canonical_RDKit.unique().tolist()
    training_set_smiles = all_smiles.SMILES_Canonical_RDKit[~all_smiles.SMILES_Canonical_RDKit.isin(validation_set_smiles)]
    return training_set_smiles

def CalculateCosineSimilarity(cls_training, cls_validation, k=10):
    cls_training = np.asarray(cls_training, dtype=np.float32)
    cls_validation = np.asarray(cls_validation, dtype=np.float32)

    # Calculate cosine similarity between validation and training set
    cosine_sim = cosine_similarity(cls_validation, cls_training)

    # Calculate average, max, and median similarity for each entry
    avg_sim = np.mean(cosine_sim, axis=1)
    max_sim = np.max(cosine_sim, axis=1)
    median_sim = np.median(cosine_sim, axis=1)

    # Calculate mean similarity to the top k neighbors for each entry
    top_k_sim = np.argpartition(cosine_sim, -k, axis=1)[:,-k:]

    # Calculate mean similarity to k closest neighbors for each entry
    mean_k_sim = np.mean(np.take_along_axis(cosine_sim, top_k_sim, axis=1), axis=1)

    return list(max_sim), list(avg_sim), list(median_sim), list(mean_k_sim)
