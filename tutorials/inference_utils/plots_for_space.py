import pandas as pd
import numpy as np
import umap
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sklearn.decomposition import PCA
from inference_utils.pytorch_data_utils import check_training_data

def __formathover(df):
    formatted_hover = ''''''
    for col in df.columns:
        formatted_hover += (col + ':<br>' + df[col].astype(str) + '<br>')
    return formatted_hover

def PlotPCA_CLSProjection(model_type, endpoint, effect, species_group, show_all_predictions, inference_df=None):

    all_preds = pd.read_pickle(f'../data/tutorials/predictions/combined_predictions_and_errors.pkl.zip', compression='zip')
    cls_df = pd.read_pickle(f'../data/tutorials/predictions/{model_type}_{species_group}_CLS_embeddings.pkl.zip', compression='zip')
    all_preds = all_preds.merge(cls_df, on='SMILES_Canonical_RDKit')
    embeddings = np.array(all_preds.CLS_embeddings.tolist()).astype(np.float32)
    # If we want to plot predicted chemicals from streamlit prediction
    if inference_df is not None:
        embeddings = np.concatenate((embeddings, inference_df.CLS_embeddings.tolist()), axis=0).astype(np.float32)

    pcomp = PCA(n_components=2)
    pcac = pcomp.fit_transform(embeddings)
    all_preds['pc1'], all_preds['pc2'] = pcac[:len(all_preds),0], pcac[:len(all_preds),1]
    if inference_df is not None:
        inference_df['pc1'], inference_df['pc2'] = pcac[len(all_preds):,0], pcac[len(all_preds):,1]

    # Get which SMILES are training
    train_effect_preds = all_preds[all_preds[f'{model_type}_{species_group}_{endpoint}_{effect} effect match'].astype(bool)]
    train_endpoint_preds = all_preds[all_preds[f'{model_type}_{species_group}_{endpoint}_{effect} endpoint match'].astype(bool)]
    remaining_preds = all_preds[(~all_preds[f'{model_type}_{species_group}_{endpoint}_{effect} effect match'].astype(bool)) | (~all_preds[f'{model_type}_{species_group}_{endpoint}_{effect} endpoint match'].astype(bool))]

    fig = make_subplots(rows=1, cols=1,
        subplot_titles=(['']),
        horizontal_spacing=0.02)

    if show_all_predictions:
        hover = __formathover(remaining_preds[['SMILES_Canonical_RDKit', f'{model_type}_{species_group} L1Error']])
        fig.add_trace(go.Scatter(x=remaining_preds.pc1, y=remaining_preds.pc2, 
                        mode='markers',
                        text=hover,
                        name='Not in training data',
                        marker=dict(colorscale='turbo_r',
                                    cmax=4,
                                    cmin=-4,
                                    color=remaining_preds[f'{model_type}_{species_group}_{endpoint}_{effect} predictions log10(mg/L)'],
                                    size=5,
                                    opacity=0.7,
                                    colorbar=dict(
                                        title='mg/L',
                                        tickvals=[2,0,-2,-4],
                                        ticktext=["10<sup>2</sup>", "10<sup>0</sup>", "10<sup>-2</sup>", "<10<sup>-4</sup>"],
                                        orientation='h'),
                        )),
                        row=1, col=1)

    hover = __formathover(train_endpoint_preds[['SMILES_Canonical_RDKit', f'{model_type}_{species_group} L1Error']])
    fig.add_trace(go.Scatter(x=train_endpoint_preds.pc1, y=train_endpoint_preds.pc2, 
                    mode='markers',
                    text=hover,
                    name='Training data: In Species group (with Endpoint match)',
                    marker=dict(colorscale='turbo_r',
                                cmax=4,
                                cmin=-4,
                                color=train_endpoint_preds[f'{model_type}_{species_group}_{endpoint}_{effect} predictions log10(mg/L)'],
                                size=5,
                                opacity=0.7,
                                line=dict(width=1.2,
                                        color='red'),
                                colorbar=dict(
                                    title='mg/L',
                                    tickvals=[2,0,-2,-4],
                                    ticktext=["10<sup>2</sup>", "10<sup>0</sup>", "10<sup>-2</sup>", "<10<sup>-4</sup>"],
                                    orientation='h'),
                    )),
                    row=1, col=1)
    
    hover = __formathover(train_effect_preds[['SMILES_Canonical_RDKit', f'{model_type}_{species_group} L1Error']])
    fig.add_trace(go.Scatter(x=train_effect_preds.pc1, y=train_effect_preds.pc2, 
                    mode='markers',
                    text=hover,
                    name='Training data: In Species group with Effect match (with Endpoint match)',
                    marker=dict(colorscale='turbo_r',
                                cmax=4,
                                cmin=-4,
                                color=train_effect_preds[f'{model_type}_{species_group}_{endpoint}_{effect} predictions log10(mg/L)'],
                                size=5,
                                opacity=0.7,
                                line=dict(width=1.2,
                                        color='black'),
                                colorbar=dict(
                                    title='mg/L',
                                    tickvals=[2,0,-2,-4],
                                    ticktext=["10<sup>2</sup>", "10<sup>0</sup>", "10<sup>-2</sup>", "<10<sup>-4</sup>"],
                                    orientation='h'),
                                )),
                    row=1, col=1)

    # Plot inferenced SMILES
    if inference_df is not None:
        hover = (inference_df['SMILES_Canonical_RDKit'])
        fig.add_trace(go.Scatter(x=inference_df.pc1, y=inference_df.pc2, 
                        mode='markers',
                        text=hover,
                        name='Predicted SMILES',
                        marker=dict(colorscale='turbo_r',
                                    cmax=4,
                                    cmin=-4,
                                    color=inference_df['predictions log10(mg/L)'],
                                    size=9,
                                    line=dict(width=2,
                                            color='black'),
                                    colorbar=dict(
                                        title='mg/L',
                                        tickvals=[2,0,-2,-4],
                                        ticktext=["10<sup>2</sup>", "10<sup>0</sup>", "10<sup>-2</sup>", "<10<sup>-4</sup>"],
                                        orientation='h'),
                                    )),
                        row=1, col=1)

    fig.update_xaxes(title_text=f"PC1 {np.round(100*pcomp.explained_variance_ratio_[0],1)}%",
        row=1, col=1)
    fig.update_yaxes(title_text=f"PC2 {np.round(100*pcomp.explained_variance_ratio_[1],1)}%",
        row=1, col=1)

    fig.update_layout(height=800)

    return fig


def PlotUMAP_CLSProjection(model_type, endpoint, effect, species_group, show_all_predictions, n_neighbors, min_dist):
    all_preds = pd.read_pickle(f'data/{model_type}_model_predictions/{species_group}_{endpoint}_{effect}_predictions.zip', compression='zip')

    embeddings = np.array(all_preds.CLS_embeddings.tolist())

    umap_model = umap.UMAP(metric = "cosine",
                      n_neighbors = n_neighbors,
                      n_components = 2,
                      low_memory = False,
                      min_dist = min_dist)
    
    umapc = umap_model.fit_transform(embeddings)
    all_preds['u1'], all_preds['u2'] = umapc[:,0], umapc[:,1]

    train_effect_preds = all_preds[all_preds['in effect training data']]
    train_species_preds = all_preds[all_preds['in species group training data']]
    remaining_preds = all_preds[(~all_preds['in effect training data']) | (~all_preds['in species group training data'])]

    fig = make_subplots(rows=1, cols=1,
        subplot_titles=(['']),
        horizontal_spacing=0.02)

    if show_all_predictions:
        hover = (remaining_preds['SMILES_Canonical_RDKit'])
        fig.add_trace(go.Scatter(x=remaining_preds.u1, y=remaining_preds.u2, 
                        mode='markers',
                        text=hover,
                        name='Not in training data',
                        marker=dict(colorscale='turbo_r',
                                    cmax=4,
                                    cmin=-4,
                                    color=remaining_preds['predictions log10(mg/L)'],
                                    size=5,
                                    colorbar=dict(
                                        title='mg/L',
                                        tickvals=[2,0,-2,-4],
                                        ticktext=["10<sup>2</sup>", "10<sup>0</sup>", "10<sup>-2</sup>", "<10<sup>-4</sup>"],
                                        orientation='h'),
                        )),
                        row=1, col=1)

    hover = (train_species_preds['SMILES_Canonical_RDKit'])
    fig.add_trace(go.Scatter(x=train_species_preds.u1, y=train_species_preds.u2, 
                    mode='markers',
                    text=hover,
                    name='Training data: In Species group',
                    marker=dict(colorscale='turbo_r',
                                cmax=4,
                                cmin=-4,
                                color=train_species_preds['predictions log10(mg/L)'],
                                size=5,
                                line=dict(width=1,
                                        color='black'),
                                colorbar=dict(
                                    title='mg/L',
                                    tickvals=[2,0,-2,-4],
                                    ticktext=["10<sup>2</sup>", "10<sup>0</sup>", "10<sup>-2</sup>", "<10<sup>-4</sup>"],
                                    orientation='h'),
                    )),
                    row=1, col=1)
    
    hover = (train_effect_preds['SMILES_Canonical_RDKit'])
    fig.add_trace(go.Scatter(x=train_effect_preds.u1, y=train_effect_preds.u2, 
                    mode='markers',
                    text=hover,
                    name='Training data: In Species group with Effect match',
                    marker=dict(colorscale='turbo_r',
                                cmax=4,
                                cmin=-4,
                                color=train_effect_preds['predictions log10(mg/L)'],
                                size=5,
                                line=dict(width=1,
                                        color='red'),
                                colorbar=dict(
                                    title='mg/L',
                                    tickvals=[2,0,-2,-4],
                                    ticktext=["10<sup>2</sup>", "10<sup>0</sup>", "10<sup>-2</sup>", "<10<sup>-4</sup>"],
                                    orientation='h'),
                                )),
                    row=1, col=1)

    fig.update_xaxes(title_text=f"UMAP1",
        row=1, col=1)
    fig.update_yaxes(title_text=f"UMAP2",
        row=1, col=1)

    fig.update_layout(height=800)

    return fig