'''
For Roman Huszar's T-maze dataset
'''
import numpy as np
import pandas as pd
from sklearn.cluster import dbscan
import pynapple as nap
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colorbar as colorbar
import matplotlib
import seaborn as sns

def get_latent_occurance_index_per_speed_level(map_latent,speed_tsd,speed_thresh_bins=[5]):
    '''
    get the indices of when a latent == MAP, divided by speed level

    map_latent: n_time, latent MAP at each time point, nap.Tsd
    speed_tsd: n_time, speed, nap.Tsd
    speed_thresh_bins: list, the speed thresholds to divide the speed into levels
    latent_occurance_index_per_speed_level: dict, key is the latent, value is a dict, key is the speed level, value is the indices of when the latent occurs
    '''
    possible_latent = np.unique(map_latent.d)
    speed_tresh_bins = np.concatenate([[0],speed_thresh_bins,[np.inf]])
    latent_occurance_index_per_speed_level = {}
    for latent_i in possible_latent:
        latent_ma = map_latent.d==latent_i
        latent_occurance_index_per_speed_level[latent_i] = {}
        for i in range(len(speed_tresh_bins)-1):
            speed_ma = (speed_tsd.d >= speed_tresh_bins[i]) & (speed_tsd.d < speed_tresh_bins[i+1])
            latent_run_ma = np.logical_and(latent_ma,speed_ma)
            latent_occurance_index_per_speed_level[latent_i][i] = np.nonzero(latent_run_ma)[0]
    return latent_occurance_index_per_speed_level


def classify_latent(map_latent,position_tsdf,speed_tsd,speed_thresh=5,min_time_bin=10):
    '''
    classify the latent into spatial and non-spatial
        spatial -- during run, one cluster
        non-spatial -- during run multi cluster, or just stationary
    map_latent: n_time, latent label, nap.Tsd
    position_tsdf: n_time, position, nap.TsdFrame
    speed_tsd: n_time, speed, nap.Tsd
    speed_thresh: speed threshold to define run
    min_time_bin: minimum time to be considered as spatial
    '''
    speed_tsd = speed_tsd.interpolate(map_latent)
    position_tsdf = position_tsdf.interpolate(map_latent)
    
    is_spatial_all_latent = {}
    cluster_label_per_time_all_latent={}
    # possible_latent = np.unique(map_latent)
    latent_occurance_index_per_speed_level = get_latent_occurance_index_per_speed_level(map_latent,speed_tsd,[speed_thresh])
    for latent_i,occurance_index_per_speed_level in latent_occurance_index_per_speed_level.items():
        
        latent_run_index=occurance_index_per_speed_level[1]
        
        if len(latent_run_index)>min_time_bin:
            tocluster=position_tsdf[latent_run_index]['x','y'].d
            core_samples, labels=dbscan(tocluster,eps=10,metric='euclidean',)
            cluster_label_per_time_all_latent[latent_i] = labels
            if set(labels)== set([-1,0]) or set(labels)== set([0]): # spatial only if one cluster /+ noise
                is_spatial_all_latent[latent_i] = True
            else: # all noise, or multi cluster
                is_spatial_all_latent[latent_i]=False
        else:
            is_spatial_all_latent[latent_i] = False
    is_spatial_all_latent=pd.Series(is_spatial_all_latent)

    spatial_latent = is_spatial_all_latent.loc[is_spatial_all_latent].index
    nonspatial_latent=is_spatial_all_latent.loc[np.logical_not(is_spatial_all_latent)].index

    latent_classify_res = {'spatial_latent':spatial_latent,'nonspatial_latent':nonspatial_latent,'is_spatial_all_latent':is_spatial_all_latent,'cluster_label_per_time_all_latent':cluster_label_per_time_all_latent,'latent_occurance_index_per_speed_level':latent_occurance_index_per_speed_level}
    return latent_classify_res

def plot_maze_background(spk_beh_df,ds=10,fig=None,ax=None,mode='line',**kwargs):
    if isinstance(spk_beh_df ,nap.TsdFrame):
        spk_beh_df = spk_beh_df.as_dataframe()
    kwargs_ = dict(c='grey',alpha=0.5)
    kwargs_.update(kwargs)
    if ax is None:
        fig,ax=plt.subplots()
    if mode=='line':
        ax.plot(spk_beh_df['x'].values[::ds],spk_beh_df['y'].values[::ds],**kwargs_)
    elif mode=='scatter':
        ax.scatter(spk_beh_df['x'].values[::ds],spk_beh_df['y'].values[::ds],s=1,**kwargs_)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    return fig,ax

from matplotlib.colors import Normalize

def plot_state_list_vs_position(state_l, map_state,behavior_tsdf,pos_col=['x','y'],fig=None,ax=None,
                                speed_col='speed_gauss',
                                speed_category_thresh = [2], # use this to categorize running and immobility
                                cmap_name='Spectral_r',
                                kwargs_scatter = dict(s=20,alpha=1),
                                marker_per_speed_category = ['^','o'],
                                do_plot_maze=False,
                                position_tsdf=None,
                                ds=10,
                                maze_c='grey',
                                maze_alpha=0.3,
                                hide_box=True,
                                seperate_colorbar=True,
                                colorbar=False,
                                background_mode='line',
                               ):
    '''
    visualize the distribution of some state as a function of 2d position
    plot running and immobility with different marker shape
    state_l: n_state, the selected state to be plotted
    map_state: n_time, ; the maximum posterior state per time
    behavior_tsdf: n_time x n_behavior_variable; has positions

    find times when one state is the MAP, plot the corresponding positions of those times
    '''

    
    
    cmap=plt.get_cmap(cmap_name)
    if ax is None:
        fig,ax=plt.subplots()
    
    if do_plot_maze:
        assert position_tsdf is not None
        plot_maze_background(position_tsdf,ds=ds,fig=fig,ax=ax,c=maze_c,alpha=maze_alpha,mode=background_mode)


    # plot running and immobility with different marker shape
    speed_category = pd.cut(behavior_tsdf['speed_gauss'],bins=[0,*speed_category_thresh,np.inf],labels=False)
    speed_category_unique = np.unique(speed_category)
    speed_category_unique = speed_category_unique[np.logical_not(np.isnan(speed_category_unique))].astype(int)
    
    
    # Plot running and immobility
    state_l_ind = np.arange(len(state_l)) 
    norm=Normalize(vmin=0,vmax=len(state_l))
    colors = cmap(norm(state_l_ind)) # color based on the index of state within state_l, not the state value
    
    # if only plotting one state, then color based on time
    # time for all time points, not just the MAP time points; this way can compare across different states and see temporal evoluation
    if len(state_l)==1:
        color_time = True
        mask = map_state==state_l[0]
        time_l_all = behavior_tsdf.t
        time_l_map = time_l_all[mask]
        norm = Normalize(vmin=time_l_all.min(),vmax=time_l_all.max())
        colors = cmap(norm(time_l_map)) 
        
    else: 
        color_time = False
        
        

    for speed_category_i in speed_category_unique:
        speed_category_mask = speed_category==speed_category_i
        s = marker_per_speed_category[speed_category_i]
        for ii,state_i in enumerate(state_l):
            mask = map_state==state_i
            mask = np.logical_and(mask, speed_category_mask)
            try:
                
                if not color_time:
                    ax.scatter(behavior_tsdf[mask][pos_col[0]].values,behavior_tsdf[mask][pos_col[1]].values,c=colors[ii],marker=s,**kwargs_scatter)
                else: # if color based on time
                    ax.scatter(behavior_tsdf[mask][pos_col[0]].values,behavior_tsdf[mask][pos_col[1]].values,edgecolors=colors,facecolors='none',marker=s,**kwargs_scatter)
                    
            except Exception as e:
                print(e)
                print('n time points: ',mask.sum())
    if hide_box:
        ax.axis('off')
    
    if color_time and colorbar: # add colorbar to plot
        plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap),ax=ax)
    to_return = fig,ax

    
     
    if seperate_colorbar: # return a seperate colorbar
        fig_cbar = plt.figure()
        ax_cbar = fig_cbar.add_axes([0.05, 0.80, 0.05, 0.9])
        
        cb = matplotlib.colorbar.ColorbarBase(ax_cbar, orientation='vertical', 
                                    cmap='Spectral_r')
        cb.set_ticks([0,1])
        if color_time:
            cb.set_ticklabels(['Start','End'])
        to_return = fig,ax,fig_cbar,cb
        

    return to_return
