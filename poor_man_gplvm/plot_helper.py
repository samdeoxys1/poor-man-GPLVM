'''
helper functions for plotting
'''

import numpy as np
import matplotlib.pyplot as plt
import pynapple as nap
import sys,os
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['svg.fonttype'] = 'none'

def save_fig(fig,fig_name,fig_dir='./figs',fig_format=['png','svg'],dpi=300):
    '''
    save figure to fig_dir
    '''
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)
    for fmt in fig_format:
        fig.savefig(os.path.join(fig_dir,fig_name+f'.{fmt}'),dpi=dpi,bbox_inches='tight')
        print(f'saved {fig_name}.{fmt} to {fig_dir}')
    plt.close(fig)

def plot_mean_error_plot(data,error_type='ci',mean_axis=0,fig=None,ax=None,**kwargs):
    '''
    plt the mean and error of the data
    data: pd.DataFrame or np.ndarray
    error_type: 'ci' or 'std'
    mean_axis: axis to take the mean of, same for error; plot the other axis
    '''
    if fig is None:
        fig,ax = plt.subplots()
    if error_type == 'ci':
        mean = np.mean(data,axis=mean_axis)
        error = np.std(data,axis=mean_axis) / np.sqrt(data.shape[mean_axis])
    elif error_type == 'std':
        mean = np.mean(data,axis=mean_axis)
        error = np.std(data,axis=mean_axis)
    else:
        raise ValueError(f'error_type {error_type} not supported')
    ax.plot(mean,**kwargs)
    if isinstance(data,pd.DataFrame):
        xs = mean.index
    else:
        xs = np.arange(len(mean))
    ax.fill_between(xs,mean-error,mean+error,alpha=0.5)
    return fig,ax
