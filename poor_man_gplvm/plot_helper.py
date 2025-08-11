'''
helper functions for plotting
'''

import numpy as np
import matplotlib.pyplot as plt
import pynapple as nap
import sys,os
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['svg.fonttype'] = 'none'

def save_fig(fig,fig_name,fig_dir='./figs',fig_format=['png','svg'],dpi=300):
    '''
    save figure to fig_dir
    '''
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)
    for fmt in fig_format:
        fig.savefig(os.path.join(fig_dir,fig_name+f'.{fmt}'),dpi=dpi)
        print(f'saved {fig_name}.{fmt} to {fig_dir}')
    plt.close(fig)