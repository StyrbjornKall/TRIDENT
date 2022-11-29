from .figure_flags import *
import pandas as pd 
import numpy as np

def UpdateFigLayout(fig, xrange, yrange, size, nfigs, legend: str='topright'):
    fig.update_xaxes(showline=True, linewidth=3, linecolor='Black')
    fig.update_yaxes(showline=True, linewidth=3, linecolor='Black')
    try:
        [fig.layout.annotations[id].font.update(size=FONTSIZE) for id in range(nfigs)]
    except:
        fig.layout.font.update(size=FONTSIZE)
    fig.update_layout(
        font_family=FONT,
        font=dict(size=FONTSIZE), 
        plot_bgcolor='rgba(0, 0, 0, 0)',
        width=size[0],
        height=size[1])

    try:
        if legend == 'topright':
            fig.update_layout(legend=dict(
                yanchor='top',
                y=0.99,
                xanchor="right",
                x=0.99
            ))
        elif legend == 'topleft':
            fig.update_layout(legend=dict(
                yanchor='top',
                y=0.99,
                xanchor="left",
                x=0.01
            ))
        elif legend == 'bottomright':
            fig.update_layout(legend=dict(
                yanchor='bottom',
                y=0.01,
                xanchor="right",
                x=0.99
            ))
        elif legend == 'bottomleft':
            fig.update_layout(legend=dict(
                yanchor='bottom',
                y=0.01,
                xanchor="left",
                x=0.01
            ))
    except:
        pass
    fig.update_xaxes(range=xrange)
    fig.update_yaxes(range=yrange)


def RescaleAxes(fig, xaxis, yaxis):
    if xaxis:
        fig.update_xaxes(tickmode = 'array',
        tickvals = np.array([-4,-3,-2,-1,0,1,2,3,4]),
        ticktext = np.array([10000,1000,100,10,1,10,100,1000,10000]))
    if yaxis:
        fig.update_yaxes(tickmode = 'array',
        tickvals = np.array([-4,-3,-2,-1,0,1,2,3,4]),
        ticktext = np.array([10000,1000,100,10,1,10,100,1000,10000]))