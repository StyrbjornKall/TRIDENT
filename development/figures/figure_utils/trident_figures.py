from .figure_flags import *
from .preprocess_data import Preprocess10x10Fold, GroupDataForPerformance, GetCanonicalSMILESForFigures
from .preprocess_qsar import LoadQSAR, PrepareQSARData, MatchQSAR
from .figure_functions import UpdateFigLayout, RescaleAxes
import pandas as pd 
import numpy as np
import json
import scipy.stats

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler



## CHEMBERTA AND LOSS FUNCTION COMPARISON (FROM SWEEP)
def PlotBaseModelLossfunResults(savepath):

    df = pd.read_pickle('../../data/results/basemodel_sweep_results_5x_CV_RDkit.zip', compression='zip')
    df['Canonical_SMILES_figures'] = df.SMILES_Canonical_RDKit.apply(lambda x: GetCanonicalSMILESForFigures(x))

    fig = go.Figure()

    models = ['seyonec/SMILES_tokenized_PubChem_shard00_160k','seyonec/PubChem10M_SMILES_BPE_450k']
    loss_funs = ['L1Loss', 'MSELoss']
    names = {
        'seyonec/SMILES_tokenized_PubChem_shard00_160k':
            {'L1Loss': ['SMILES Tokenizer 1M MAE Loss', '#fdcc8a'], 'MSELoss': ['SMILES Tokenizer 1M MSE Loss', '#fc8d59']},
        'seyonec/PubChem10M_SMILES_BPE_450k':
            {'L1Loss': ['BPE Tokenizer 10M MAE Loss', '#9bb9e8'], 'MSELoss': ['BPE Tokenizer 10M MSE Loss','#567fbf']}}
    xgroups = ['Mean','Median', 'Weighted Mean', 'Weighted Median']
    for model in models:
        for loss in loss_funs:
            results = df[(df['loss_fun']==loss) & (df['base_model']==model)]
            L1Error = abs(results[f'residuals'])
            mean_L1, median_L1, se_mean, MAD = (
            L1Error.mean(), 
            L1Error.median(), 
            L1Error.std()/np.sqrt(len(L1Error)), 
            abs(L1Error-L1Error.median()).median()/np.sqrt(len(L1Error)))
            results = GroupDataForPerformance(results) # Uncomment if you want weighted Avg
            L1Error = abs(results[f'residuals'])
            wmean_L1, wmedian_L1, wse_mean, wMAD = (
            L1Error.mean(), 
            L1Error.median(), 
            L1Error.std()/np.sqrt(len(L1Error)), 
            abs(L1Error-L1Error.median()).median()/np.sqrt(len(L1Error)))
            res = np.array([mean_L1, median_L1, wmean_L1, wmedian_L1])
            std = np.array([se_mean, MAD, wse_mean, wMAD])
            fig.add_trace(go.Bar(
                    name=f'{names[model][loss][0]}', 
                    x=xgroups, y=10**res,
                    error_y=dict(type='data',symmetric=False,
                        array=10**res*(10**std-1),
                        arrayminus=10**res*(1-10**-std)),
                    marker_color=names[model][loss][1],
                    marker=dict(
                            line_width=1,
                            line_color='Black'),
                ))

    fig.update_yaxes(title_text='Absolute Prediction Error (fold change)', tickfont = dict(size=FONTSIZE))

    fig.update_layout(barmode='group')
    UpdateFigLayout(fig, None, [1,10],[1000,700],1, 'topright')
    RescaleAxes(fig, False, False)

    fig.show(renderer='png')

    if savepath != None:
        fig.write_image(savepath+'.png', scale=7)
        fig.write_image(savepath+'.svg')
        fig.write_html(savepath+'.html')


## RESIDUAL HISTOGRAM ONE PER SMILES
def PlotKFoldResidualHistUsingWAvgPreds(savepath, name, endpoint, species_group):

    predictions = GroupDataForPerformance(Preprocess10x10Fold(name=name))
    residuals = predictions.residuals

    fig = make_subplots(rows=1, cols=1,
    subplot_titles=(['Residual distribution']),
    column_widths=[1], row_heights=[1], shared_yaxes=False,
        horizontal_spacing=0.02)

    fig.add_trace(go.Histogram(
        x=residuals,
        xbins=dict(
            size=0.04
        ),
        marker_color=colors[endpoint],
        marker=dict(line_width=1,
            line_color='Black')
    ), row=1, col=1)

    fig.add_vline(x=3, line_dash="dot", annotation_text=f"{np.round(100*sum(residuals>3)/len(residuals),1)}%", annotation_font_size=FONTSIZE, row=1, col=1)
    fig.add_vline(x=2, line_dash="dot", annotation_text=f"{np.round(100*sum(residuals>2)/len(residuals),1)}%", annotation_font_size=FONTSIZE, row=1, col=1)
    fig.add_vline(x=1, line_dash="dot", annotation_text=f"{np.round(100*sum(residuals>1)/len(residuals),1)}%", annotation_font_size=FONTSIZE, row=1, col=1)
    fig.add_vline(x=-3, line_dash="dot", annotation_text=f"{np.round(100*sum(residuals<-3)/len(residuals),1)}%", annotation_font_size=FONTSIZE, row=1, col=1, annotation_position="top left")
    fig.add_vline(x=-2, line_dash="dot", annotation_text=f"{np.round(100*sum(residuals<-2)/len(residuals),1)}%", annotation_font_size=FONTSIZE, row=1, col=1, annotation_position="top left")
    fig.add_vline(x=-1, line_dash="dot", annotation_text=f"{np.round(100*sum(residuals<-1)/len(residuals),1)}%", annotation_font_size=FONTSIZE, row=1, col=1, annotation_position="top left")

    print(f'''{sum(residuals.abs()>3)} >3
    {sum(residuals.abs()>2)} >2
    {sum(residuals.abs()>1)} >1''')

    fig.update_layout(
        yaxis_title_text='Count',
        xaxis_title_text="Prediction Error (fold change)",
        bargap=0.02, 
        showlegend=False
    )

    UpdateFigLayout(fig, [-4,4], None,[1000,600],1)
    RescaleAxes(fig, True, False)

    fig.show(renderer='png')
    if savepath != None:
        fig.write_image(savepath+'.png', scale=7)
        fig.write_image(savepath+'.svg')
        fig.write_html(savepath+'.html')

    
    if endpoint == 'EC50EC10':
        del fig
        ec50_predictions = GroupDataForPerformance(Preprocess10x10Fold(name=f'EC50_{species_group}'))
        ec10_predictions = GroupDataForPerformance(Preprocess10x10Fold(name=f'EC10_{species_group}'))
        combo_predictions = GroupDataForPerformance(Preprocess10x10Fold(name=name))
        
        ec50residuals = predictions[(combo_predictions.endpoint=='EC50') & (combo_predictions.Canonical_SMILES_figures.isin(ec50_predictions.Canonical_SMILES_figures))].residuals
        ec10residuals = predictions[(combo_predictions.endpoint=='EC10') & (combo_predictions.Canonical_SMILES_figures.isin(ec10_predictions.Canonical_SMILES_figures))].residuals

        fig = make_subplots(rows=1, cols=1,
        subplot_titles=(['Residual distribution']),
        column_widths=[1], row_heights=[1], shared_yaxes=False,
            horizontal_spacing=0.02)

        fig.add_trace(go.Histogram(
            x=ec50residuals,
            xbins=dict(
                size=0.04
            ),
            marker_color=colors[endpoint],
            marker=dict(line_width=1,
                line_color='Black')
        ), row=1, col=1)

        fig.add_vline(x=3, line_dash="dot", annotation_text=f"{np.round(100*sum(ec50residuals>3)/len(ec50residuals),1)}%", annotation_font_size=FONTSIZE, row=1, col=1)
        fig.add_vline(x=2, line_dash="dot", annotation_text=f"{np.round(100*sum(ec50residuals>2)/len(ec50residuals),1)}%", annotation_font_size=FONTSIZE, row=1, col=1)
        fig.add_vline(x=1, line_dash="dot", annotation_text=f"{np.round(100*sum(ec50residuals>1)/len(ec50residuals),1)}%", annotation_font_size=FONTSIZE, row=1, col=1)
        fig.add_vline(x=-3, line_dash="dot", annotation_text=f"{np.round(100*sum(ec50residuals<-3)/len(ec50residuals),1)}%", annotation_font_size=FONTSIZE, row=1, col=1, annotation_position="top left")
        fig.add_vline(x=-2, line_dash="dot", annotation_text=f"{np.round(100*sum(ec50residuals<-2)/len(ec50residuals),1)}%", annotation_font_size=FONTSIZE, row=1, col=1, annotation_position="top left")
        fig.add_vline(x=-1, line_dash="dot", annotation_text=f"{np.round(100*sum(ec50residuals<-1)/len(ec50residuals),1)}%", annotation_font_size=FONTSIZE, row=1, col=1, annotation_position="top left")

        fig.update_layout(
            yaxis_title_text='Count',
            xaxis_title_text="Prediction Error (fold change)",
            bargap=0.02, 
            showlegend=False
        )

        UpdateFigLayout(fig, [-4,4], None,[1000,600],1)
        RescaleAxes(fig, True, False)

        fig.show(renderer='png')
        if savepath != None:
            fig.write_image(savepath+'_EC50.png', scale=7)
            fig.write_image(savepath+'_EC50.svg')
            fig.write_html(savepath+'_EC50.html')

        del fig
        fig = make_subplots(rows=1, cols=1,
        subplot_titles=(['Residual distribution']),
        column_widths=[1], row_heights=[1], shared_yaxes=False,
            horizontal_spacing=0.02)

        fig.add_trace(go.Histogram(
            x=ec10residuals,
            xbins=dict(
                size=0.04
            ),
            marker_color=colors[endpoint],
            marker=dict(line_width=1,
                line_color='Black')
        ), row=1, col=1)

        fig.add_vline(x=3, line_dash="dot", annotation_text=f"{np.round(100*sum(ec10residuals>3)/len(ec10residuals),1)}%", annotation_font_size=FONTSIZE, row=1, col=1)
        fig.add_vline(x=2, line_dash="dot", annotation_text=f"{np.round(100*sum(ec10residuals>2)/len(ec10residuals),1)}%", annotation_font_size=FONTSIZE, row=1, col=1)
        fig.add_vline(x=1, line_dash="dot", annotation_text=f"{np.round(100*sum(ec10residuals>1)/len(ec10residuals),1)}%", annotation_font_size=FONTSIZE, row=1, col=1)
        fig.add_vline(x=-3, line_dash="dot", annotation_text=f"{np.round(100*sum(ec10residuals<-3)/len(ec10residuals),1)}%", annotation_font_size=FONTSIZE, row=1, col=1, annotation_position="top left")
        fig.add_vline(x=-2, line_dash="dot", annotation_text=f"{np.round(100*sum(ec10residuals<-2)/len(ec10residuals),1)}%", annotation_font_size=FONTSIZE, row=1, col=1, annotation_position="top left")
        fig.add_vline(x=-1, line_dash="dot", annotation_text=f"{np.round(100*sum(ec10residuals<-1)/len(ec10residuals),1)}%", annotation_font_size=FONTSIZE, row=1, col=1, annotation_position="top left")

        fig.update_layout(
            yaxis_title_text='Count',
            xaxis_title_text="Prediction Error (fold change)",
            bargap=0.02, 
            showlegend=False
        )

        UpdateFigLayout(fig, [-4,4], None,[1000,600],1)
        RescaleAxes(fig, True, False)

        fig.show(renderer='png')
        if savepath != None:
            fig.write_image(savepath+'_EC10.png', scale=7)
            fig.write_image(savepath+'_EC10.svg')
            fig.write_html(savepath+'_EC10.html')


## BARPLOT PERFORMANCE SINGLE MODELS (M50, M10)
def PlotKFoldSingleBarUsingWAvgPreds(savepath, ec50_name, ec10_name):

    errors=['Median', 'Mean']
    names = [ec50_name,ec10_name]
    colnames = ['EC50','EC10']
    fig = go.Figure()

    for i, name in enumerate(names):
        predictions = GroupDataForPerformance(Preprocess10x10Fold(name=name, uselogdata=True))
        _, mean_L1, median_L1, se, MAD = predictions.residuals, predictions.L1error.mean(), predictions.L1error.median(), predictions.L1error.sem(), (abs(predictions.L1error-predictions.L1error.median())).median()/np.sqrt(len(predictions))

        fig.add_trace(go.Bar(
            name=name,
            x=errors, y=[10**median_L1, 10**mean_L1],
            error_y=dict(type='data',
            symmetric=False,
            array=[10**median_L1*(10**MAD-1),10**mean_L1*(10**se-1)],
            arrayminus=[10**median_L1*(1-10**-MAD), 10**mean_L1*(1-10**-se)]),
            marker_color=colors[colnames[i]],
            marker=dict(line_width=1,
            line_color='Black')
        ))

    fig.update_layout(title='Mean and Median Absolute Error 10x10 fold CV')

    fig.update_yaxes(title_text="Absolute Prediction Error (fold change)")

    fig.update_layout(barmode='group')
    UpdateFigLayout(fig, None, [1,8],[1000,700],1, 'topright')

    
    fig.show(renderer='png')

    if savepath != None:
        fig.write_image(savepath+'.png', scale=7)
        fig.write_image(savepath+'.svg')
        fig.write_html(savepath+'.html')

## BARPLOT PERFORMANCE COMBOMODEL (M50/10)
def PlotKFoldComboBarUsingWAvgPreds(savepath, combomodel, species_group):

    fig = go.Figure()

    endpoints = [f'EC50',f'EC10']
    xgroups=[['Median', 'Mean', 'Median', 'Mean'], 
        ['EC50','EC50','EC10','EC10']]
    qsarbarcolors = [colors['EC50'],colors['EC50'],colors['EC10'],colors['EC10']]

    qsar = []
    qsarse = []
    ecotoxformer = []
    ecotoxformerse = []
    for endpoint in endpoints:
        single_predictions = GroupDataForPerformance(Preprocess10x10Fold(name=endpoint+f'_{species_group}'))
        _, mean_L1, median_L1, se, MAD = single_predictions.residuals, single_predictions.L1error.mean(), single_predictions.L1error.median(), single_predictions.L1error.sem(), (abs(single_predictions.L1error-single_predictions.L1error.median())).median()/np.sqrt(len(single_predictions))
        qsar += [median_L1, mean_L1]
        qsarse += [MAD, se]
        combo_predictions = GroupDataForPerformance(Preprocess10x10Fold(name=combomodel))
        combo_predictions = combo_predictions[(combo_predictions.endpoint==endpoint) & (combo_predictions.Canonical_SMILES_figures.isin(single_predictions.Canonical_SMILES_figures))]
        _, mean_L1, median_L1, se, MAD = combo_predictions.residuals, combo_predictions.L1error.mean(), combo_predictions.L1error.median(), combo_predictions.L1error.sem(), (abs(combo_predictions.L1error-combo_predictions.L1error.median())).median()/np.sqrt(len(combo_predictions))
        ecotoxformer += [median_L1, mean_L1]
        ecotoxformerse += [MAD, se]

    qsar = np.array(qsar)
    qsarse = np.array(qsarse)
    ecotoxformer = np.array(ecotoxformer)
    ecotoxformerse = np.array(ecotoxformerse)
    fig.add_trace(go.Bar(
                name=endpoint+f'_{species_group}', x=xgroups, y=10**qsar,
                error_y=dict(type='data',symmetric=False,
                array=10**qsar*(10**qsarse-1),
                arrayminus=10**qsar*(1-10**-qsarse)),
                marker_color = qsarbarcolors, 
                marker=dict(
                        line_width=1,
                        line_color='Black'),
            ))

    fig.add_trace(go.Bar(
            name=f'Combined EC50&EC10', 
            x=xgroups, y=10**ecotoxformer,
            error_y=dict(type='data',symmetric=False,
                array=10**ecotoxformer*(10**ecotoxformerse-1),
                arrayminus=10**ecotoxformer*(1-10**-ecotoxformerse)),
            marker_color = colors['EC50EC10'],
            marker=dict(
                    line_width=1,
                    line_color='Black'),
        ))
    
    fig.update_layout(title='Mean & Median AE 10x10 fold CV')
    fig.update_yaxes(title_text="Absolute Prediction Error (fold change)")    
    UpdateFigLayout(fig, None, [1,8],[1000,700],1, 'topright')

    fig.update_layout(barmode='group')
    fig.show(renderer='png')

    if savepath != None:
        fig.write_image(savepath+'.png', scale=7)
        fig.write_image(savepath+'.svg')
        fig.write_html(savepath+'.html')
    

## PRINCIPAL COMPONENT ANALYSIS (PCA)
def PlotPCA_CLSProjection(savepath, endpoint, species_group, flipxaxis, flipyaxis):

    results = pd.read_pickle(f'../../data/results/{endpoint}_{species_group}_final_model_training_data_RDkit.zip', compression='zip')
    results['Canonical_SMILES_figures'] = results.SMILES_Canonical_RDKit.apply(lambda x: GetCanonicalSMILESForFigures(x))
    results['labels'] = results['mgperL']
    CLS_dict = dict(zip(results.Canonical_SMILES_figures, results.CLS_embeddings.tolist()))

    results.drop(columns=['CLS_embeddings'], inplace=True)

    results = GroupDataForPerformance(results)
    results = results.groupby(['Canonical_SMILES_figures','cmpdname','endpoint'], as_index=False).median(numeric_only=True)
    results['L1Error'] = results.residuals.abs()
    results['CLS_embeddings'] = results.Canonical_SMILES_figures.apply(lambda x: CLS_dict[x])

    embeddings = np.array(results.CLS_embeddings.tolist())
    #embeddings_scaled = StandardScaler().fit_transform(embeddings)

    pcomp = PCA(n_components=3)
    pca = pd.DataFrame(data = pcomp.fit_transform(embeddings), columns = ['pc1', 'pc2','pc3'])
    results = pd.concat([results, pca], axis=1)


    if (endpoint == 'EC50') or (endpoint == 'EC10'):

        hover = (results['Canonical_SMILES_figures']+'<br>'+results['cmpdname'])

        fig = make_subplots(rows=1, cols=1,
        subplot_titles=(['CLS embedding 2D PCA projection']),
        horizontal_spacing=0.02)
        # for EC10 or EC50
        if flipxaxis:
            x = results.pc1*-1
        else:
            x = results.pc1
        if flipyaxis:
            y = results.pc2*-1
        else:
            y = results.pc2
        fig.add_trace(go.Scatter(x=x, y=y, 
                        mode='markers',
                        text=hover,
                        marker=dict(colorscale=[(0, '#67000d'),
                         (0.25, '#fb6a4a'),
                          (0.5, '#c994c7'),
                          (0.75, '#4393c3'), 
                          (1, '#023858')],
                                    cmax=3,
                                    cmin=-4,
                                    color=results.labels,
                                    size=7,
                                    colorbar=dict(
                                        title='mg/L',
                                        tickvals=[2,0,-2,-4],
                                        ticktext=["10<sup>2</sup>", "10<sup>0</sup>", "10<sup>-2</sup>", "<10<sup>-4</sup>"]),
                                    )),
                        row=1, col=1)

    else:
        fig = make_subplots(rows=1, cols=2,
        subplot_titles=(['CLS embedding 2D PCA projection']),
        horizontal_spacing=0.02)
        # Combo model
        ec50 = results[results.endpoint=='EC50']
        ec10 = results[results.endpoint=='EC10']
        hover50 = (ec50['Canonical_SMILES_figures']+'<br>'+ec50['cmpdname'])
        hover10 = (ec10['Canonical_SMILES_figures']+'<br>'+ec10['cmpdname'])
        fig.add_trace(go.Scatter(x=ec50.pc1, y=ec50.pc2, 
                        mode='markers',
                        text=hover50,
                        name='EC50',
                        marker=dict(colorscale='Viridis',
                                    cmax=max(ec50.labels),
                                    cmin=min(ec50.labels),
                                    color=ec50.labels,
                                    size=5,
                                    colorbar=dict(title=''),
                                    line_width=1,
                                    line_color='Black')),
                        row=1, col=1)

        fig.add_trace(go.Scatter(x=ec10.pc1, y=ec10.pc2, 
                        mode='markers',
                        text=hover10,
                        name='EC10',
                        marker=dict(colorscale='Inferno',
                                    cmax=max(ec10.labels),
                                    cmin=min(ec10.labels),
                                    color=ec10.labels,
                                    size=5,
                                    colorbar=dict(title='', x = 1.15),
                                    line_width=1,
                                    line_color='Black')),
                        row=1, col=2)

    fig.update_xaxes(title_text=f"PC1 {np.round(100*pcomp.explained_variance_ratio_[0],1)}%",
        row=1, col=1)
    fig.update_yaxes(title_text=f"PC2 {np.round(100*pcomp.explained_variance_ratio_[1],1)}%",
        row=1, col=1)

    UpdateFigLayout(fig,None, None, [1000,800],1)
    if endpoint == 'EC50EC10':
        UpdateFigLayout(fig,None, None, [2000,800],1)
        fig.update_layout(legend=dict(
            yanchor="bottom",
            y=0.01,
            xanchor="left",
            x=0.01
        ))
    
    fig.show(renderer = 'png')

    if savepath != None:
        fig.write_image(savepath+'.png', scale=7)
        fig.write_image(savepath+'.svg')
        fig.write_html(savepath+'.html')


## QSAR COMPARISON INTERSECTION OF APPLICABILITY DOMAINS
def PlotQSARcompBarUsingWAvgPredsInterersect(savepath, predictions, endpoint, species_group):

    fig = go.Figure()
    xgroups  = ['Median','Mean']
    QSARs = ['ECOSAR','VEGA','TEST']

    if endpoint == 'EC50':
        if species_group != 'algae':
            intersect = predictions[~((predictions.ECOSAR.isna() | predictions.VEGA.isna()) | predictions.TEST.isna())]
        else:
            intersect = predictions[~((predictions.ECOSAR.isna() | predictions.VEGA.isna()))]
        print(intersect.shape)
    else:
        intersect = predictions[~(predictions.ECOSAR.isna() | predictions.VEGA.isna())]
        print(intersect.shape)
    L1Error = abs(intersect.residuals)
    mean_L1_model, median_L1_model, se_model, MAD_model = L1Error.mean(), L1Error.median(), L1Error.sem(), (abs(L1Error-L1Error.median())).median()/np.sqrt(len(L1Error))

    fig.add_trace(go.Bar(
            name=f'TRIDENT', x=xgroups, y=[10**median_L1_model, 10**mean_L1_model],
            error_y=dict(type='data',
            symmetric=False,
            array=[10**median_L1_model*(10**MAD_model-1),10**mean_L1_model*(10**se_model-1)],
            arrayminus=[10**median_L1_model*(1-10**-MAD_model), 10**mean_L1_model*(1-10**-se_model)]),
            marker_color = colors[endpoint],
            marker=dict(
                    line_width=1,
                    line_color='Black'),
        ))

    for i, qsar_tool in enumerate(QSARs):
        try:
            L1Error = abs(intersect[f'{qsar_tool}_residuals'])
        except:
            L1Error = pd.DataFrame([None], columns=['L1Error'])
        mean_L1_qsar, median_L1_qsar, se_qsar, MAD_qsar = L1Error.mean(), L1Error.median(), L1Error.sem(), (abs(L1Error-L1Error.median())).median()/np.sqrt(len(L1Error))
            
        fig.add_trace(go.Bar(
            name=f'{qsar_tool}', x=xgroups, y=[10**median_L1_qsar, 10**mean_L1_qsar],
            error_y=dict(type='data',
            symmetric=False,
            array=[10**median_L1_qsar*(10**MAD_qsar-1),10**mean_L1_qsar*(10**se_qsar-1)],
            arrayminus=[10**median_L1_qsar*(1-10**-MAD_qsar), 10**mean_L1_qsar*(1-10**-se_qsar)]),
            marker_color = colors[qsar_tool],
            marker=dict(
                    line_width=1,
                    line_color='Black'),
        ))

    fig.update_layout(title=f'Mean & Median AE 10x10 fold CV QSAR intersection n={len(intersect)}')
    fig.update_yaxes(title_text="Absolute Prediction Error (fold change)")
    UpdateFigLayout(fig, None, [1,28],[1000,700],1)
    fig.update_yaxes(tick0=1)
    fig.update_layout(barmode='group')
    fig.show(renderer='png')

    if savepath != None:
        fig.write_image(savepath+'.png', scale=7)
        fig.write_image(savepath+'.svg')
        fig.write_html(savepath+'.html')


## QSAR COMPARISON RESIDUAL SCATTER PLOT
def PlotQSARresidualScatter(savepath, predictions, endpoint):

    models = ['TRIDENT', 'ECOSAR','VEGA','TEST']

    rows = [1,1,2,2]
    cols = [1,2,1,2]
    fig = make_subplots(rows=2, cols=2,
    subplot_titles=(
    f'TRIDENT n={len(predictions)}', 
    f'ECOSAR n={len(predictions["ECOSAR"].dropna())}', 
    f'VEGA n={len(predictions["VEGA"].dropna())}',
    f'T.E.S.T. n={len(predictions["TEST"].dropna())}'), shared_yaxes=True, shared_xaxes=True,
    horizontal_spacing=0.05, vertical_spacing=0.1)

    for i, model in enumerate(models):
        tmp_df = predictions[~predictions[model].isna()]

        if model != 'TRIDENT':
            residuals = tmp_df[f'{model}_residuals']
        else:
            residuals = tmp_df['residuals']
        AE = abs(residuals)
        if model=='TRIDENT':
            model=endpoint
        fig.add_trace(go.Scatter(x=tmp_df.labels, y=AE, 
                        mode='markers',
                        marker=dict(
                                line_width=1,
                                line_color='Black'),
                        marker_color=colors[model], showlegend=False),
                        row=rows[i], col=cols[i])

        try:
            fig.add_hline(y=3, line_dash="dot", annotation_text=f"{np.round(100*sum(AE>3)/len(AE),1)}%", annotation_font_size=FONTSIZE, annotation_bgcolor='White', row=rows[i], col=cols[i])
            fig.add_hline(y=2, line_dash="dot", annotation_text=f"{np.round(100*sum(AE>2)/len(AE),1)}%", annotation_font_size=FONTSIZE, annotation_bgcolor='White',row=rows[i], col=cols[i])
            fig.add_hline(y=1, line_dash="dot", annotation_text=f"{np.round(100*sum(AE>1)/len(AE),1)}%", annotation_font_size=FONTSIZE, annotation_bgcolor='White', row=rows[i], col=cols[i])
            print(f'{model} >1 {sum(AE>1)}')
            print(f'{model} >2 {sum(AE>2)}')
            print(f'{model} >3 {sum(AE>3)}')
        except:
            pass

    fig.update_yaxes(title_text="Absolute Prediction Error (fold change)", row=1, col=1)
    fig.update_yaxes(title_text="Absolute Prediction Error (fold change)",row=2, col=1)
    fig.update_xaxes(title='Actual Concentration (mg/L)', row=2,col=1)
    fig.update_xaxes(title='Actual Concentration (mg/L)', row=2,col=2)
    fig.update_yaxes(dtick=1)
    UpdateFigLayout(fig, [-4,4], [-0.2,4],[1000,1000],4)
    RescaleAxes(fig, False, True)

    fig.show(renderer='png')

    if savepath != None:
        fig.write_image(savepath+'.png', scale=7)
        fig.write_image(savepath+'.svg')
        fig.write_html(savepath+'.html')



## QSAR COMPARISON RESIDUAL SCATTER PLOT
def PlotQSARresidualScatterIntersect(savepath, predictions, endpoint):

    models = ['TRIDENT', 'ECOSAR','VEGA','TEST']

    rows = [1,1,2,2]
    cols = [1,2,1,2]

    if endpoint == 'EC50':
        predictions = predictions[~((predictions.ECOSAR.isna() | predictions.VEGA.isna()) | predictions.TEST.isna())]
        print(predictions.shape)
    else:
        predictions = predictions[~(predictions.ECOSAR.isna() | predictions.VEGA.isna())]
        print(predictions.shape)
    L1Error = abs(predictions.residuals)

    fig = make_subplots(rows=2, cols=2,
    subplot_titles=(
    f'TRIDENT n={len(predictions)}', 
    f'ECOSAR n={len(predictions["ECOSAR"].dropna())}', 
    f'VEGA n={len(predictions["VEGA"].dropna())}',
    f'T.E.S.T. n={len(predictions["TEST"].dropna())}'), shared_yaxes=True, shared_xaxes=True,
    horizontal_spacing=0.05, vertical_spacing=0.1)

    for i, model in enumerate(models):
        tmp_df = predictions[~predictions[model].isna()]

        if model != 'TRIDENT':
            residuals = tmp_df[f'{model}_residuals']
        else:
            residuals = tmp_df['residuals']
        AE = abs(residuals)
        if model=='TRIDENT':
            model=endpoint
        fig.add_trace(go.Scatter(x=tmp_df.labels, y=AE, 
                        mode='markers',
                        marker=dict(
                                line_width=1,
                                line_color='Black'),
                        marker_color=colors[model], showlegend=False),
                        row=rows[i], col=cols[i])

        try:
            fig.add_hline(y=3, line_dash="dot", annotation_text=f"{np.round(100*sum(AE>3)/len(AE),1)}%", annotation_font_size=FONTSIZE, annotation_bgcolor='White', row=rows[i], col=cols[i])
            fig.add_hline(y=2, line_dash="dot", annotation_text=f"{np.round(100*sum(AE>2)/len(AE),1)}%", annotation_font_size=FONTSIZE, annotation_bgcolor='White',row=rows[i], col=cols[i])
            fig.add_hline(y=1, line_dash="dot", annotation_text=f"{np.round(100*sum(AE>1)/len(AE),1)}%", annotation_font_size=FONTSIZE, annotation_bgcolor='White', row=rows[i], col=cols[i])
            print(f'{model} >1 {sum(AE>1)}')
            print(f'{model} >2 {sum(AE>2)}')
            print(f'{model} >3 {sum(AE>3)}')
        except:
            pass

    fig.update_yaxes(title_text="Absolute Prediction Error (fold change)", row=1, col=1)
    fig.update_yaxes(title_text="Absolute Prediction Error (fold change)",row=2, col=1)
    fig.update_xaxes(title='Actual Concentration (mg/L)', row=2,col=1)
    fig.update_xaxes(title='Actual Concentration (mg/L)', row=2,col=2)
    fig.update_yaxes(dtick=1)
    UpdateFigLayout(fig, [-4,4], [-0.2,4],[1000,1000],4)
    RescaleAxes(fig, False, True)

    fig.show(renderer='png')

    if savepath != None:
        fig.write_image(savepath+'.png', scale=7)
        fig.write_image(savepath+'.svg')
        fig.write_html(savepath+'.html')


## QSAR COMPARISON PREDICTIONS VS. LABELS SCATTER
def PlotQSARcompScatter(savepath, predictions, endpoint):
    models = ['TRIDENT', 'ECOSAR','VEGA','TEST']

    rows = [1,1,2,2]
    cols = [1,2,1,2]
    fig = make_subplots(rows=2, cols=2,
    subplot_titles=(
    f'TRIDENT n={len(predictions)}', 
    f'ECOSAR n={len(predictions["ECOSAR"].dropna())}', 
    f'VEGA n={len(predictions["VEGA"].dropna())}',
    f'T.E.S.T. n={len(predictions["TEST"].dropna())}'), shared_yaxes=True, shared_xaxes=True,
    horizontal_spacing=0.05, vertical_spacing=0.1)


    for i, model in enumerate(models):
        tmp_df = predictions[~predictions[model].isna()]
        X = tmp_df.labels.to_numpy()
        Y = tmp_df[model].to_numpy()
        try:
            slope, intercept, r_value, _, _ = scipy.stats.linregress(X, Y)
            y_pred = slope*np.array([-4,4])+intercept
            line = ([-4,4], y_pred, r_value**2)
        except:
            line = ([-4,4],None,np.NaN)
        preds = tmp_df[model]
        if model=='TRIDENT':
            model=endpoint

        fig.add_trace(go.Scatter(x=tmp_df.labels, y=preds, 
                        mode='markers',
                        marker=dict(
                                line_width=1,
                                line_color='Black'),
                        marker_color=colors[model], showlegend=False),
                        row=rows[i], col=cols[i])
        fig.add_trace(go.Scatter(
                        x=line[0],
                        y=line[1],
                        mode='lines+text',
                        marker_color='Black',
                        line={'dash': 'dot'},
                        text=['',f"R2 {line[2]:.2f}"],
                        showlegend=False,
                        textposition='bottom left'
                        ), rows[i], cols[i])
        

    fig.update_yaxes(title_text="Absolute Prediction Error (fold change)", row=1, col=1)
    fig.update_yaxes(title_text="Absolute Prediction Error (fold change)",row=2, col=1)
    fig.update_xaxes(title='Actual Concentration (mg/L)', row=2,col=1)
    fig.update_xaxes(title='Actual Concentration (mg/L)', row=2,col=2)
    fig.update_yaxes(dtick=1)
    UpdateFigLayout(fig, [-4,4], [-4,4],[1000,1000],4)
    RescaleAxes(fig, False, False)

    fig.show(renderer='png')

    if savepath != None:
        fig.write_image(savepath+'.png', scale=7)
        fig.write_image(savepath+'.svg')
        fig.write_html(savepath+'.html')



## QSAR COVERAGE/APPLICABILITY (INCLUDING EXPERIMENTAL/TRAINING) DATA
def PlotQSARCoverageComboBar(savepath, inside_AD, species_group):
    if inside_AD:
        AD='AD'
    else:
        AD='allpreds'

    names = ['TRIDENT','ECOSAR','VEGA','TEST']
    barcolorsec50 = [colors['EC50'],colors['ECOSAR'],colors['VEGA'], colors['TEST']]
    barcolorsec10 = [colors['EC10'],colors['ECOSAR'],colors['VEGA'], colors['TEST']]
    fig = go.Figure()

    for endpoint in ['EC50', 'EC10']:
        ECOSAR, VEGA, TEST = LoadQSAR(endpoint=endpoint, species_group=species_group) 
        ECOSAR_tmp, TEST_tmp, VEGA_tmp = PrepareQSARData(ECOSAR, TEST, VEGA, inside_AD=inside_AD, remove_experimental=False, species_group=species_group)
        avg_predictions = Preprocess10x10Fold(endpoint+f"_{species_group}")
        all_smiles = avg_predictions.Canonical_SMILES_figures.drop_duplicates().tolist()

        testsmiles = TEST_tmp.Canonical_SMILES_figures[TEST_tmp.Canonical_SMILES_figures.isin(all_smiles)]
        ecosmiles = ECOSAR_tmp.Canonical_SMILES_figures[ECOSAR_tmp.Canonical_SMILES_figures.isin(all_smiles)]
        vegasmiles = VEGA_tmp.Canonical_SMILES_figures[VEGA_tmp.Canonical_SMILES_figures.isin(all_smiles)]

        avg_predictions.Canonical_SMILES_figures.drop_duplicates().to_csv(f'../../development/figures/figures_for_publication/venn/all_smiles_{endpoint+f"_{species_group}"}_for_venn.txt', header=None, index=None, sep='\n', mode='w')
        ecosmiles.drop_duplicates().to_csv(f'../../development/figures/figures_for_publication/venn/ecosar_smiles_{endpoint+f"_{species_group}"}_{AD}_for_venn.txt', header=None, index=None, sep='\n', mode='w')
        testsmiles.drop_duplicates().to_csv(f'../../development/figures/figures_for_publication/venn/test_smiles_{endpoint+f"_{species_group}"}_{AD}_for_venn.txt',  header=None, index=None, sep='\n', mode='w')
        vegasmiles.drop_duplicates().to_csv(f'../../development/figures/figures_for_publication/venn/vega_smiles_{endpoint+f"_{species_group}"}_{AD}_for_venn.txt',  header=None, index=None, sep='\n', mode='w')

    
        models = [all_smiles, ecosmiles, vegasmiles, testsmiles]

        coverage = [100*(len(set(model) & set(all_smiles)))/len(all_smiles) for model in models]

        if endpoint=='EC50':
            fig.add_trace(go.Bar(
                name=endpoint+f"_{species_group}",
                x=names, y=coverage,
                marker_color = barcolorsec50,
                marker=dict(
                        line_width=1,
                        line_color='Black'),
            ))
        else:
            fig.add_trace(go.Bar(
                name=endpoint+f"_{species_group}",
                x=names, y=coverage,
                marker_color = barcolorsec10,
                marker=dict(
                        line_width=1,
                        line_color='Black'),
                marker_pattern_shape="x"
            ))

    UpdateFigLayout(fig, None, [0,105],[1000,700],1)
    fig.update_yaxes(title_text="Coverage (%)")
    
    fig.show(renderer='png')

    if savepath != None:
        fig.write_image(savepath+'.png', scale=7)
        fig.write_image(savepath+'.svg')
        fig.write_html(savepath+'.html')  


def PlotQSARComp3inOne(savepath, endpoint,inside_AD, use_weighted_avg):

    algae, algae_wavg = GetQSARPredictionForSpecies(endpoint+'_'+'algae', endpoint=endpoint, species_group='algae', durations=['short exposure','medium exposure', 'long exposure'], inside_AD=inside_AD)
    invertebrates, invertebrates_wavg = GetQSARPredictionForSpecies(endpoint+'_'+'invertebrates', endpoint=endpoint, species_group='invertebrates', durations=['short exposure','medium exposure', 'long exposure'], inside_AD=inside_AD)
    fish, fish_wavg = GetQSARPredictionForSpecies(endpoint+'_'+'fish', endpoint=endpoint, species_group='fish', durations=['short exposure','medium exposure', 'long exposure'], inside_AD=inside_AD)

    if use_weighted_avg:
        fish = fish_wavg
        algae = algae_wavg
        invertebrates = invertebrates_wavg

    colors_specific = {'ECOSAR': colors['ECOSAR'], 'VEGA': colors['VEGA'], 'TEST': colors['TEST']}
    if endpoint == 'EC50':
        colors_specific['fish'] = '#e0ecf4'
        colors_specific['invertebrates'] = '#fff7bc'
        colors_specific['algae'] = '#c2e699'
    else:
        colors_specific['fish'] = '#9ebcda'
        colors_specific['invertebrates'] = '#fec44f'
        colors_specific['algae'] = '#78c679'

    colorpergroup = {
        'TRIDENT': [colors_specific['fish'], colors_specific['invertebrates'],colors_specific['algae']],
        'ECOSAR': [colors_specific['ECOSAR'], colors_specific['ECOSAR'], colors_specific['ECOSAR']],
        'VEGA': [colors_specific['VEGA'], colors_specific['VEGA'], colors_specific['VEGA']],
        'TEST': [colors_specific['TEST'], colors_specific['TEST'], colors_specific['TEST']]}

    # Get intersect
    if endpoint == 'EC50':
        fish = fish[~((fish.ECOSAR.isna() | fish.VEGA.isna()) | fish.TEST.isna())]
        invertebrates = invertebrates[~((invertebrates.ECOSAR.isna() | invertebrates.VEGA.isna()) | invertebrates.TEST.isna())]
        algae = algae[~(algae.ECOSAR.isna() | algae.VEGA.isna())] 
    if endpoint == 'EC10':
        fish = fish[~(fish.ECOSAR.isna() | fish.VEGA.isna())]
        invertebrates = invertebrates[~(invertebrates.ECOSAR.isna() | invertebrates.VEGA.isna())]
        algae = algae[~(algae.ECOSAR.isna() | algae.VEGA.isna())] 

    fig = go.Figure()

    xgroups = ['fish','invertebrates', 'algae']
    QSARs = ['TRIDENT','ECOSAR','VEGA','TEST']

    for i, qsar_tool in enumerate(QSARs):
        
        try:
            fish_L1Error = abs(fish[f'{qsar_tool}_residuals'])
            invert_L1Error = abs(invertebrates[f'{qsar_tool}_residuals'])
        except:
            fish_L1Error = pd.DataFrame([None], columns=['L1Error'])
            invert_L1Error = pd.DataFrame([None], columns=['L1Error'])

        try:
            alg_L1Error = abs(algae[f'{qsar_tool}_residuals'])
        except:
            alg_L1Error = pd.DataFrame([None], columns=['L1Error'])

        _, fish_median_L1_qsar, _, fish_MAD_qsar = fish_L1Error.mean(), fish_L1Error.median(), fish_L1Error.sem(), (abs(fish_L1Error-fish_L1Error.median())).median()/np.sqrt(len(fish_L1Error))
        _, invert_median_L1_qsar, _, invert_MAD_qsar = invert_L1Error.mean(), invert_L1Error.median(), invert_L1Error.sem(), (abs(invert_L1Error-invert_L1Error.median())).median()/np.sqrt(len(invert_L1Error))
        _, alg_median_L1_qsar, _, alg_MAD_qsar = alg_L1Error.mean(), alg_L1Error.median(), alg_L1Error.sem(), (abs(alg_L1Error-alg_L1Error.median())).median()/np.sqrt(len(alg_L1Error))

        fig.add_trace(go.Bar(
            name=f'{qsar_tool}', x=xgroups, y=[10**fish_median_L1_qsar, 10**invert_median_L1_qsar, 10**alg_median_L1_qsar],
            error_y=dict(type='data',
            symmetric=False,
            array=[10**fish_median_L1_qsar*(10**fish_MAD_qsar-1),10**invert_median_L1_qsar*(10**invert_MAD_qsar-1),10**alg_median_L1_qsar*(10**alg_MAD_qsar-1)],
            arrayminus=[10**fish_median_L1_qsar*(1-10**-fish_MAD_qsar), 10**invert_median_L1_qsar*(1-10**-invert_MAD_qsar), 10**alg_median_L1_qsar*(1-10**-alg_MAD_qsar)]),
            marker_color = colorpergroup[qsar_tool],
            marker=dict(
                    line_width=1,
                    line_color='Black'),
        ))

    print(f'''
    SMILES in fish: {len(set(fish.Canonical_SMILES_figures))}
    SMILES in invertebrates: {len(set(invertebrates.Canonical_SMILES_figures))}
    SMILES in algae: {len(set(algae.Canonical_SMILES_figures))}''')

    fig.update_yaxes(title_text='Absolute Prediction Error (fold change)', tickfont = dict(size=FONTSIZE))

    fig.update_layout(barmode='group')
    UpdateFigLayout(fig, None, [1,28],[1000,700],1, 'topright')
    RescaleAxes(fig, False, False)

    fig.show(renderer='png')

    if savepath != None:
        fig.write_image(savepath+'.png', scale=7)
        fig.write_image(savepath+'.svg')
        fig.write_html(savepath+'.html')


def GetQSARPredictionForSpecies(name, endpoint, species_group, durations, inside_AD):
    avg_predictions = Preprocess10x10Fold(name=name)
    ECOSAR, VEGA, TEST = LoadQSAR(endpoint=endpoint, species_group=species_group)
    ECOSAR, TEST, VEGA = PrepareQSARData(ECOSAR, TEST, VEGA, inside_AD=inside_AD, remove_experimental=True, species_group=species_group)
    TEST_dict = dict(zip(TEST.Canonical_SMILES_figures, TEST.value))
    VEGA_dict = dict(zip(VEGA.Canonical_SMILES_figures, VEGA.value))
    ECOSAR_dict = dict(zip(ECOSAR.Canonical_SMILES_figures, ECOSAR.value))

    qsar_preds = MatchQSAR(avg_predictions, ECOSAR_dict, TEST_dict, VEGA_dict, endpoint=endpoint, duration=durations, species_group=species_group)
    qsar_preds[['ECOSAR_residuals','VEGA_residuals','TEST_residuals']] = qsar_preds[['ECOSAR','VEGA','TEST']] - qsar_preds[['labels']].to_numpy()
    weighted_avg_qsar_preds = GroupDataForPerformance(qsar_preds)
    if (endpoint == 'EC10') | (species_group=='algae'):
        weighted_avg_qsar_preds[['TEST', 'TEST_residuals']] = None

    return qsar_preds, weighted_avg_qsar_preds