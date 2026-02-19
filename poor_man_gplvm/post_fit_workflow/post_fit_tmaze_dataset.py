'''
For Roman Huszar's T-maze dataset
'''
import numpy as np
import pandas as pd
import os
import time
import json
from sklearn.cluster import dbscan
import pynapple as nap
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colorbar as colorbar
import matplotlib
import seaborn as sns
import poor_man_gplvm.plot_helper as ph
import tqdm
import poor_man_gplvm.analysis_helper as ah
import poor_man_gplvm.verify_latent_jump as vlj
import poor_man_gplvm.post_fit_workflow.get_event_windows as gew
import poor_man_gplvm.trial_analysis as tri
import poor_man_gplvm.post_fit_workflow.behavior_classification_tmaze as bct
import poor_man_gplvm.transition_analysis as trans
import poor_man_gplvm.post_fit_workflow.shuffle_test_decoding as shuf

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
from scipy.spatial.distance import cdist
def get_dist_to_maze(xy_l,xy_sampled_all):
    '''
    idea, first find closest sample points, then find the segment and line direction of them, 
    then use the closest sample points as reference to compute projection distance

    '''
    dist = np.min(cdist(xy_l,xy_sampled_all),axis=1) # dist computed as shorted dist to sample
    
    return dist

def classify_latent(map_latent,position_tsdf,speed_tsd,tmaze_xy_sampled_all,speed_thresh=5,dist_to_maze_thresh=5,min_total_time=30,min_run_time=10,min_off_maze_time=10,eps=3):
    '''
    classsify into: spatial-running, immobility, off-maze
        
        spatial-running: during high speed, more time bin than min_time_bin
        immobility: not enough running time; 
        off-maze: certain amount of high speed time outside the maze
        get rid of off-maze from spatial-running
        
    so if both spatial and immobility, count as spatial; this is a generous defitiion for spatial but stringent for non-spatial
    there will be edge cases, ignore for now (e.g. some off maze, some immobility; sporadic; ignore for now)
    time threshold here are in time bin same as map_latent, usually 100ms, need to adjust accordingly
    '''
    speed_tsd = speed_tsd.interpolate(map_latent)
    # maze_coord_df,xy_sampled_all = preprt.get_tmaze_xy_sample(position_tsdf,place_bin_size=1.,do_plot=False)
    position_tsdf = position_tsdf.interpolate(map_latent)
    
    
    is_spatial_all_latent = {}
    is_immobility_all_latent = {}
    is_off_maze_all_latent = {}
    is_low_occurence_all_latent = {}
    cluster_label_per_time_all_latent={}
    latent_total_time_all_latent={}
    
    # possible_latent = np.unique(map_latent)
    latent_occurance_index_per_speed_level = get_latent_occurance_index_per_speed_level(map_latent,speed_tsd,[speed_thresh])
    for latent_i,occurance_index_per_speed_level in latent_occurance_index_per_speed_level.items():
        latent_run_index=occurance_index_per_speed_level[1]
        latent_immobility_index=occurance_index_per_speed_level[0]
        latent_total_time = len(latent_run_index) + len(latent_immobility_index)
        latent_total_time_all_latent[latent_i] = latent_total_time
        
        latent_immobility_fraction = (len(latent_immobility_index) / (len(latent_immobility_index) + len(latent_run_index)))
        
        
        is_immobility_all_latent[latent_i] = False
        if len(latent_run_index)>min_run_time:
            is_spatial_all_latent[latent_i] = True

        else:
            is_spatial_all_latent[latent_i] = False
            is_immobility_all_latent[latent_i] = True
            is_off_maze_all_latent[latent_i] = False
        if len(latent_run_index)>0:
            xy_l = position_tsdf[latent_run_index]['x','y'].d
            dist_to_maze=get_dist_to_maze(xy_l,tmaze_xy_sampled_all)
            n_off_maze_time = (dist_to_maze >dist_to_maze_thresh).sum()
            if n_off_maze_time > min_off_maze_time:
                is_off_maze_all_latent[latent_i] = True
                is_spatial_all_latent[latent_i] = False
            else:
                is_off_maze_all_latent[latent_i] = False
        
        if is_spatial_all_latent[latent_i]:
            tocluster=position_tsdf[latent_run_index]['x','y'].d
            core_samples, labels=dbscan(tocluster,eps=eps,metric='euclidean',)
            cluster_label_per_time_all_latent[latent_i] = labels
        

    is_spatial_all_latent=pd.Series(is_spatial_all_latent)
    is_immobility_all_latent=pd.Series(is_immobility_all_latent)
    is_off_maze_all_latent=pd.Series(is_off_maze_all_latent)
    spatial_latent = is_spatial_all_latent.loc[is_spatial_all_latent].index
    immobility_latent = is_immobility_all_latent.loc[is_immobility_all_latent].index
    off_maze_latent = is_off_maze_all_latent.loc[is_off_maze_all_latent].index
    nonspatial_latent=is_spatial_all_latent.loc[np.logical_not(is_spatial_all_latent)].index
    cateogry_all_latent=np.zeros(len(is_spatial_all_latent),dtype=object)
    cateogry_all_latent[is_spatial_all_latent] = 'spatial'
    cateogry_all_latent[is_immobility_all_latent] = 'immobility'
    cateogry_all_latent[is_off_maze_all_latent] = 'off_maze'

    latent_classify_res = {'spatial_latent':spatial_latent,'nonspatial_latent':nonspatial_latent,'immobility_latent':immobility_latent,'off_maze_latent':off_maze_latent,'is_spatial_all_latent':is_spatial_all_latent,'is_immobility_all_latent':is_immobility_all_latent,'is_off_maze_all_latent':is_off_maze_all_latent,'latent_occurance_index_per_speed_level':latent_occurance_index_per_speed_level,'cateogry_all_latent':cateogry_all_latent,'latent_total_time_all_latent':latent_total_time_all_latent,'cluster_label_per_time_all_latent':cluster_label_per_time_all_latent}
    return latent_classify_res




# def classify_latent(map_latent,position_tsdf,speed_tsd,speed_thresh=5,min_time_bin=10,eps=1):
#     '''
#     classify the latent into spatial and non-spatial
#         spatial -- during run, one cluster
#         non-spatial -- during run multi cluster, or just stationary
#     map_latent: n_time, latent label, nap.Tsd
#     position_tsdf: n_time, position, nap.TsdFrame
#     speed_tsd: n_time, speed, nap.Tsd
#     speed_thresh: speed threshold to define run
#     min_time_bin: minimum time to be considered as spatial
#     '''
#     speed_tsd = speed_tsd.interpolate(map_latent)
#     position_tsdf = position_tsdf.interpolate(map_latent)
    
#     is_spatial_all_latent = {}
#     cluster_label_per_time_all_latent={}
#     # possible_latent = np.unique(map_latent)
#     latent_occurance_index_per_speed_level = get_latent_occurance_index_per_speed_level(map_latent,speed_tsd,[speed_thresh])
#     for latent_i,occurance_index_per_speed_level in latent_occurance_index_per_speed_level.items():
        
#         latent_run_index=occurance_index_per_speed_level[1]
        
#         if len(latent_run_index)>min_time_bin:
#             tocluster=position_tsdf[latent_run_index]['x','y'].d
#             core_samples, labels=dbscan(tocluster,eps=eps,metric='euclidean',)
#             cluster_label_per_time_all_latent[latent_i] = labels
#             if set(labels)== set([-1,0]) or set(labels)== set([0]): # spatial only if one cluster /+ noise
#                 is_spatial_all_latent[latent_i] = True
#             else: # all noise, or multi cluster
#                 is_spatial_all_latent[latent_i]=False
#         else:
#             is_spatial_all_latent[latent_i] = False
#     is_spatial_all_latent=pd.Series(is_spatial_all_latent)

#     spatial_latent = is_spatial_all_latent.loc[is_spatial_all_latent].index
#     nonspatial_latent=is_spatial_all_latent.loc[np.logical_not(is_spatial_all_latent)].index

#     latent_classify_res = {'spatial_latent':spatial_latent,'nonspatial_latent':nonspatial_latent,'is_spatial_all_latent':is_spatial_all_latent,'cluster_label_per_time_all_latent':cluster_label_per_time_all_latent,'latent_occurance_index_per_speed_level':latent_occurance_index_per_speed_level}
#     return latent_classify_res

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

def plot_latent_list_vs_position(latent_l, map_latent,behavior_tsdf,pos_col=['x','y'],fig=None,ax=None,
                                speed_col='speed_gauss',
                                speed_category_thresh = [5], # use this to categorize running and immobility
                                cmap_name='Spectral_r',
                                kwargs_scatter = dict(s=20,alpha=1),
                                marker_per_speed_category = ['^','o'],
                                do_plot_maze=False,
                                position_tsdf=None,
                                ds=5, # downsample for maze plot
                                maze_c='grey',
                                maze_alpha=0.3,
                                maze_zorder=0,
                                hide_box=True,
                                seperate_colorbar=True,
                                colorbar=False,
                                background_mode='line',
                                color_time=True,
                                color='red',
                               ):
    '''
    this plots one plot, can be multiple latents or single latent
    visualize the distribution of some latent as a function of 2d position
    plot running and immobility with different marker shape
    latent_l: n_latent, the selected latent to be plotted
    map_latent: n_time, ; the maximum posterior latent per time
    behavior_tsdf: n_time x n_behavior_variable; has positions

    find times when one latent is the MAP, plot the corresponding positions of those times
    '''

    if isinstance(map_latent,nap.Tsd):
        map_latent = map_latent.d
    
    cmap=plt.get_cmap(cmap_name)
    if ax is None:
        fig,ax=plt.subplots()
    
    if do_plot_maze:
        assert position_tsdf is not None
        plot_maze_background(position_tsdf,ds=ds,fig=fig,ax=ax,c=maze_c,alpha=maze_alpha,mode=background_mode,zorder=maze_zorder)


    # plot running and immobility with different marker shape
    speed_category = pd.cut(behavior_tsdf[speed_col],bins=[0,*speed_category_thresh,np.inf],labels=False)
    speed_category_unique = np.unique(speed_category)
    speed_category_unique = speed_category_unique[np.logical_not(np.isnan(speed_category_unique))].astype(int)
    
    
    # Plot running and immobility
    latent_l_ind = np.arange(len(latent_l)) 
    norm=Normalize(vmin=0,vmax=len(latent_l))
    if color is None and len(latent_l_ind)>1: 
        colors = cmap(norm(latent_l_ind)) # color based on the index of latent within latent_l, not the latent value
    else:
        colors = color
    # if only plotting one latent, then color based on time
    # time for all time points, not just the MAP time points; this way can compare across different latents and see temporal evoluation
    if len(latent_l)==1 and color_time:
        
        mask = map_latent==latent_l[0]
        time_l_all = behavior_tsdf.t
        time_l_map = time_l_all[mask]
        norm = Normalize(vmin=time_l_all.min(),vmax=time_l_all.max())
        colors = cmap(norm(time_l_map)) 
        
        
        

    for speed_category_i in speed_category_unique:
        speed_category_mask = speed_category==speed_category_i
        s = marker_per_speed_category[speed_category_i]
        for ii,latent_i in enumerate(latent_l):
            mask = map_latent==latent_i
            mask = np.logical_and(mask, speed_category_mask)
            try:
                
                if not color_time:
                    # ax.scatter(behavior_tsdf[mask][pos_col[0]].values,behavior_tsdf[mask][pos_col[1]].values,c=colors[ii],marker=s,**kwargs_scatter)
                    ax.scatter(behavior_tsdf[mask][pos_col[0]].values,behavior_tsdf[mask][pos_col[1]].values,edgecolors=colors[ii],facecolors='none',marker=s,**kwargs_scatter)
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

def get_latent_field_properties(latent_occurance_index_per_speed_level,cluster_label_per_time_all_latent,position_label,trial_intervals=None,trial_range_to_compare={'early':(2,12),'late':(-11,-1)}, do_circular_stat=False):
    '''
    get field center, width, change in trial, etc.

    position_label: can be 'lin' from position_tsdf, or ['x','y']; the position label whose summary statistics we need
    latent_occurance_index_per_speed_level: dict, key is the latent, value is a dict, key is the speed level (assume 1 is running), value is the indices of when the latent occurs
    cluster_label_per_time_all_latent: dict, key is the latent, value is an array of cluster labels for each running time point; the cluster 0 is what we care about, since we assume the spatial latent has one field

    assume the indices are aligned for position_label and latent_occurance_index_per_speed_level
    '''
    trials_sub_k = {}
    
    

    # Pre-compute bounds from full position_label if doing circular stats
    if do_circular_stat:
        data_min, data_max = _compute_data_bounds_for_circular(position_label)
    
    if trial_intervals is not None:
        for k,val in trial_range_to_compare.items():
            trials_sub = trial_intervals[val[0]:val[1]]
            trials_sub_k[k] = trials_sub

    properties_d_all = {}
    for latent_i,occurance_index_per_speed_level in latent_occurance_index_per_speed_level.items():
        properties_d ={}
        if latent_i in cluster_label_per_time_all_latent: # to be clustered the occurences must be above some number, in classify_latent, this is set to 10
            for cluster_i in np.unique(cluster_label_per_time_all_latent[latent_i]):
                if cluster_i==-1:
                    continue
                # import pdb; pdb.set_trace()
                time_sel=occurance_index_per_speed_level[1][cluster_label_per_time_all_latent[latent_i]==cluster_i] # 1 selects ruing; 

                position_sub = position_label[time_sel]
                if not do_circular_stat:
                    mean = position_sub.mean(axis=0)
                    std = position_sub.std(axis=0)
                else:
                    # Assume 1D when circular stats are requested, use bounds from full position_label
                    mean = _circular_mean(position_sub, data_min, data_max)
                    std = _circular_std(position_sub, data_min, data_max)
                properties_d['mean'] = mean
                properties_d['std'] = std
                properties_d['n_occurance'] = len(time_sel)

                if trial_intervals is not None:
                    position_mean_sub_trials = {}
                    for k,trials_sub in trials_sub_k.items():
                        if not do_circular_stat:
                            position_mean_sub_trials[k] = position_sub.restrict(trials_sub).mean()
                        else:
                            # Use same bounds for all trial-restricted subsets
                            position_mean_sub_trials[k] = _circular_mean(position_sub.restrict(trials_sub), data_min, data_max)
                        properties_d[f'{k}_mean'] = position_mean_sub_trials[k]
                    if not do_circular_stat:
                        position_mean_sub_trials['diff']=position_mean_sub_trials['late'] - position_mean_sub_trials['early']
                    else:
                        # Use same bounds for circular difference
                        position_mean_sub_trials['diff'] = _circular_diff(position_mean_sub_trials['late'], position_mean_sub_trials['early'], data_min, data_max)
                    properties_d[f'diff'] = position_mean_sub_trials['diff']
                
                properties_d = pd.Series(properties_d)
                
                properties_d_all[latent_i,cluster_i] = properties_d
    properties_d_all = pd.DataFrame(properties_d_all).T # (latent, cluster) x properties
    return properties_d_all

# Helper functions for circular statistics on a linear variable wrapped between data_min and data_max
def _extract_values_1d(obj):
    """Return a 1D numpy array of data values from supported inputs (nap objects, pandas, numpy)."""
    if hasattr(obj, 'd'):
        vals = obj.d
    elif isinstance(obj, (pd.Series, np.ndarray, list)):
        vals = np.asarray(obj)
    else:
        # Fallback: try numpy conversion
        vals = np.asarray(obj)
    # Squeeze to 1D if possible (assuming circular stats are only for 1D)
    return np.ravel(vals)

def _compute_data_bounds_for_circular(variable_obj):
    vals_all = _extract_values_1d(variable_obj)
    data_min_ = np.nanmin(vals_all)
    data_max_ = np.nanmax(vals_all)
    return data_min_, data_max_

def _circular_mean(values_1d, a, b):
    vals = _extract_values_1d(values_1d)
    if vals.size == 0:
        return np.nan
    period = b - a
    if not np.isfinite(period) or period == 0:
        return np.nan
    theta = 2 * np.pi * (vals - a) / period
    C = np.nanmean(np.cos(theta))
    S = np.nanmean(np.sin(theta))
    if not np.isfinite(C) or not np.isfinite(S):
        return np.nan
    mean_ang = np.arctan2(S, C) % (2 * np.pi)
    mean_val = a + (period * mean_ang) / (2 * np.pi)
    return mean_val

def _circular_std(values_1d, a, b):
    vals = _extract_values_1d(values_1d)
    if vals.size == 0:
        return np.nan
    period = b - a
    if not np.isfinite(period) or period == 0:
        return np.nan
    theta = 2 * np.pi * (vals - a) / period
    C = np.nanmean(np.cos(theta))
    S = np.nanmean(np.sin(theta))
    R = np.hypot(C, S)
    if R <= 0 or not np.isfinite(R):
        return np.nan
    std_rad = np.sqrt(-2 * np.log(R))
    std_val = std_rad * period / (2 * np.pi)
    return std_val

def _circular_diff(late_val, early_val, a, b):
    period = b - a
    if not np.isfinite(period) or period == 0:
        return np.nan
    # Convert to angles
    late_ang = 2 * np.pi * (late_val - a) / period
    early_ang = 2 * np.pi * (early_val - a) / period
    # Wrap difference to [-pi, pi]
    d_ang = (late_ang - early_ang + np.pi) % (2 * np.pi) - np.pi
    return d_ang * period / (2 * np.pi)


def get_latent_in_position_range(latent_occurance_index_per_speed_level,position_label,trial_intervals,reward_lin_range=(109,113),speed_level = 0,correct_only=True):
    '''
    loop over latent, get the fraction of occurance and total occurance count in some position range (lin; e.g. reward); only look in the immobility/low speed state
    latent_occurance_index_per_speed_level: from classify_latent
    position_label: 'lin' from position_tsdf
    trial_intervals: nap.IntervalSet, trial information including correctness ("choice") and visited arm ("visitedArm"), from preprocessing 
    reward_lin_range: the position range to look for occurance
    speed_level: the speed level to look for occurance

    the counting is seperated for left and right trials
    '''
    if correct_only:
        trial_intervals_correct=trial_intervals[(trial_intervals['choice']==1)] # only look at correct trials
    else:
        trial_intervals_correct=  trial_intervals
    gpb=trial_intervals_correct.groupby('visitedArm')
    intv_d = {0:trial_intervals_correct[gpb[0]],1:trial_intervals_correct[gpb[1]]}
    
    occurance_in_range_alllatent={}
    for li, occurance_per_speed_level in latent_occurance_index_per_speed_level.items():
        frac_in_range_d = {}
        total_in_range_d = {}
        for lr,intv in intv_d.items():
            oneside_trial_pos = position_label[occurance_per_speed_level[speed_level]].restrict(intv)
            if len(oneside_trial_pos)>0:
                ma = (oneside_trial_pos.d>=reward_lin_range[0]) & (oneside_trial_pos.d<=reward_lin_range[1])
                frac_in_range = ma.mean()
                total_in_range= ma.sum()
            else:
                frac_in_range=0
                total_in_range=0
            frac_in_range_d[lr] = frac_in_range
            total_in_range_d[lr] = total_in_range
        occurance_in_range = {'frac':frac_in_range_d,'total':total_in_range_d}
        occurance_in_range = pd.DataFrame(occurance_in_range)
        occurance_in_range['frac_lr_total'] = occurance_in_range['total'] / occurance_in_range['total'].sum() # fraction of occurance among left right combined, to select latent where left and right are more even
        occurance_in_range_alllatent[li] = occurance_in_range
    occurance_in_range_alllatent = pd.concat(occurance_in_range_alllatent)
    return occurance_in_range_alllatent

def get_single_reward_latent(occurance_in_range_alllatent,frac_thresh=0.7,total_thresh=10):
    '''
    filter out latent that is tuned to reward location on one side 
    occurance_in_range_alllatent: from get_latent_in_position_range, pd.dataframe, (latent, left/right) x ['frac','total','frac_lr_total']
    '''
    tuned_to_single_reward = []
    gpb=occurance_in_range_alllatent.groupby(level=0)
    for k,val in gpb:
        majority = (val['frac']>frac_thresh).sum()==1
        enough_occurance = (val['total'][val['frac']>frac_thresh] > total_thresh).all()
        if majority and enough_occurance:
            tuned_to_single_reward.append(k)
    print(tuned_to_single_reward)
    return tuned_to_single_reward

def get_both_reward_latent(occurance_in_range_alllatent,frac_thresh=0.7,total_thresh=10):
    '''
    occurance_in_range_alllatent: from get_latent_in_position_range, pd.dataframe, (latent, left/right) x ['frac','total','frac_lr_total']
    '''
    tuned_to_both_reward = []
    gpb=occurance_in_range_alllatent.groupby(level=0)
    for k,val in gpb:
        majority = (val['frac']>frac_thresh).sum()==2
        enough_occurance = (val['total'][val['frac']>frac_thresh] > total_thresh).all()
        if majority and enough_occurance:
            tuned_to_both_reward.append(k)
    print(tuned_to_both_reward)
    return tuned_to_both_reward
        
def plot_multiple_latent_spatial_map(latent_ind_l,posterior_latent_map,behavior_tsdf,position_tsdf=None,speed_thresh=5,color_time=True,kwargs_scatter = dict(s=10,alpha=0.5),color=None,speed_col='speed_gauss'):
    nplots = len(latent_ind_l)
    fig,axs=ph.subplots_wrapper(nplots,)
    if position_tsdf is None:
        position_tsdf = behavior_tsdf[['x','y']]
    for ii,i in enumerate(latent_ind_l):
        ax=axs.ravel()[ii]
        # state_l = np.arange(10)
        latent_l =[i]
        to_return=plot_latent_list_vs_position(latent_l, posterior_latent_map,behavior_tsdf,pos_col=['x','y'],fig=fig,ax=ax,
                                        speed_col=speed_col,
                                        speed_category_thresh = [speed_thresh], # use this to categorize running and immobility
                                        cmap_name='Spectral_r',
                                        kwargs_scatter = kwargs_scatter,
                                        marker_per_speed_category = ['^','o'],
                                        do_plot_maze=True,
                                        position_tsdf=position_tsdf,ds=5,
                                                seperate_colorbar=False,
                                                color_time=color_time,
                                                color=color
                                    )
        ax=to_return[1]
        ax.set_title(f'latent {i}')
    return fig,axs

def plot_multiple_latent_posterior_in_time(posterior_latent,**kwargs):
    
    nplots=posterior_latent.shape[1]
    fig,axs=ph.subplots_wrapper(nplots,**kwargs)
    for latent_ind in range(nplots):
        ax=axs.ravel()[latent_ind]
        ax.plot(posterior_latent[:,latent_ind])
        ax.set_title(f'latent {latent_ind}')
    plt.tight_layout()

    return fig,axs


def get_time_of_arrival_based_one_position(position_tsdf,lin_range=(109,113)):
    lin_sub=position_tsdf['lin'].threshold(lin_range[0],method='aboveequal').threshold(lin_range[1],method='belowequal')
    arrival_times =nap.Ts(t=lin_sub.time_support[:,0])
    return arrival_times


    
    

def find_all_index_per_latent_pair(latent_pair_l,posterior_latent_map,merge_latent_threshold=1):
    '''
    find the index within posterior_latent_map where the pre index is pair[0] and index is pair[1] (i.e. index = jump index)
    give some wiggle room if merge_latent_threshold > 0 
    return: array of array of indices
    '''
    t_l =None
    if isinstance(posterior_latent_map,nap.Tsd):
        t_l = posterior_latent_map.t
        posterior_latent_map = posterior_latent_map.d
    
    ind_l = []
    for pair in latent_pair_l:
        pre_satisfy = np.abs(posterior_latent_map[:-1]-pair[0]) <= merge_latent_threshold
        post_satisfy = np.abs(posterior_latent_map[1:]-pair[1]) <= merge_latent_threshold
        ind = np.nonzero(np.logical_and(pre_satisfy,post_satisfy))[0]
        ind = ind + 1
        ind_l.append(ind)
    ind_l=np.array(ind_l,dtype=object)
    if t_l is not None:
        ind_ts_l = np.array([nap.Ts(t_l[ind.astype(int)]) for ind in ind_l],dtype=object)
    else:
        ind_ts_l=None

    return ind_l,ind_ts_l



def find_transition_times(behavior_tsdf_aligned, trial_intervals, lin_pt=115, 
                         transition_type='arrival', tolerance=10):
    """Find transition times when animal crosses a linear position threshold."""
    transition_ts = []
    
    for tr_win in trial_intervals:
        lin_per_trial = behavior_tsdf_aligned['lin'].restrict(tr_win)
        
        pre_cross = (lin_per_trial.d <= lin_pt) & (lin_per_trial.d >= lin_pt - tolerance)
        post_cross = (lin_per_trial.d > lin_pt) & (lin_per_trial.d <= lin_pt + tolerance)
        
        transition_indices = np.nonzero(pre_cross[:-1] & post_cross[1:])[0]
        if len(transition_indices) == 0:
            continue
            
        if transition_type == 'arrival':
            transition_ind = transition_indices[0]  # first
        elif transition_type == 'departure':
            transition_ind = transition_indices[-1]  # last
            
        transition_time = lin_per_trial.t[transition_ind]
        transition_ts.append(transition_time)
    
    return nap.Ts(transition_ts)


def compute_consensus_fractions_by_window(peri_transition_matrix, max_window_size=10):
    """Compute fraction of transitions with consensus for different window sizes."""
    mid_ind = peri_transition_matrix.shape[0] // 2
    frac_d = {}
    
    for win_size_int in range(1, max_window_size + 1):
        frac = peri_transition_matrix[mid_ind-win_size_int:mid_ind+win_size_int].any(axis=0).mean()
        frac_d[win_size_int] = frac
        
    return pd.Series(frac_d)


def compute_shuffle_consensus_fractions(jump_binary_consensus, transition_ts, win=1, 
                                      win_size_int=1, n_shuffle=1000):
    """Compute shuffle control for consensus fractions around transitions."""
    frac_sh_l = []
    
    for i in tqdm.trange(n_shuffle):
        shift = np.random.randint(0, len(jump_binary_consensus))
        jump_binary_consensus_sh = np.roll(jump_binary_consensus, shift)
        peri_transition_has_jump_consensus_sh = nap.compute_perievent_continuous(
            jump_binary_consensus_sh, transition_ts, win)
        
        mid_ind = peri_transition_has_jump_consensus_sh.shape[0] // 2
        frac_sh = peri_transition_has_jump_consensus_sh[mid_ind-win_size_int:mid_ind+win_size_int].any(axis=0).mean()
        frac_sh_l.append(frac_sh)
        
    return frac_sh_l


def analyze_peri_transition_jump_consensus(behavior_tsdf_aligned, trial_intervals, jump_binary_consensus,
                               lin_pt=115, transition_type='arrival', win=1, max_window_size=10, n_shuffle=100):
    """Complete analysis of jump consensus around behavioral transitions."""
    '''
    win: window size in second for peri event
    max_window_size: max bin number to sweep; window for combining the resulting peri event
    '''
    
    # Find transition times
    transition_ts = find_transition_times(behavior_tsdf_aligned, trial_intervals, 
                                        lin_pt, transition_type)
    
    # Compute peri-transition consensus
    peri_transition_has_jump_consensus = nap.compute_perievent_continuous(
        jump_binary_consensus, transition_ts, win)
    
    # Compute consensus fractions by window size
    frac_d = compute_consensus_fractions_by_window(peri_transition_has_jump_consensus, max_window_size)
    
    # Compute shuffle controls for each window size
    shuffle_fractions = {}
    for win_size in range(1, max_window_size + 1):
        print(f"Computing shuffles for window size {win_size}...")
        shuffle_fractions[win_size] = compute_shuffle_consensus_fractions(
            jump_binary_consensus, transition_ts, win, win_size, n_shuffle)
    shuffle_fractions = pd.DataFrame(shuffle_fractions)
    
    return {
        'transition_ts': transition_ts,
        'peri_transition_matrix': peri_transition_has_jump_consensus,
        'consensus_fractions': frac_d,
        'shuffle_fractions': shuffle_fractions
    }


def latent_jump_triggered_analysis(posterior_latent_map,behavior_tsdf,spk_mat,tuning_fit,
                                     t=None,seq=None,latent_distance_thresh=1,peri_event_win=2,cols=None,contrast_axis_latent_window=0):
    '''
    for a given jump, find all occurences, get peri event of some features (behavior and contrastive axis projection)
    either give seq or t
    t is the time in second for selecting the jump
    seq is the (pre_latent,post_latent) pair
    behavior_tsdf: nap.TsdFrame, features to do peri event on
    '''
    
    if t is None:
        assert seq is not None
    else:
        postjump_ind = posterior_latent_map.get_slice(t).start
        prejump_ind = postjump_ind - 1
        seq = posterior_latent_map.d[prejump_ind:postjump_ind+1]
    
    # find occurences of the sequence
    seq_occurence_t,seq_occurence_ind = ah.get_sequence_occurence(seq,posterior_latent_map,latent_distance_thresh=latent_distance_thresh)


    if cols is None:
        cols = behavior_tsdf.columns
    peri_event_d = {}
    for col in cols:
        peri_event_d[col] = nap.compute_perievent_continuous(behavior_tsdf[col],seq_occurence_t,peri_event_win)
    
    # get the peri event of contrastive axis projection
    proj,contrast_axis=vlj.get_contrast_axis_and_proj(spk_mat,tuning_fit,seq[0],seq[1],map_state_win=contrast_axis_latent_window)
    peri_event_d['contrastive_projection'] = nap.compute_perievent_continuous(proj,seq_occurence_t,peri_event_win)

    return peri_event_d,seq_occurence_t
        
def get_null_contrastive_projection(spk_mat,tuning_fit,posterior_latent_map,jump_p_all_chain,jump_p_thresh=0.1,contrast_axis_latent_window=0,n_shuffle=100,peri_event_win=2,latent_distance_thresh=1):
    '''
    spk_mat: n_time x n_neuron
    tuning_fit: n_latent x n_neuron
    jump_p_all_chain: either n_time x n_chain or n_time; if has n_chain dimension, then exclude times when any chain is above threshold
    '''
    
    
    
    
    if jump_p_all_chain.ndim == 1:
        jump_p_all_chain = jump_p_all_chain.reshape(-1,1)
    n_chain = jump_p_all_chain.shape[1]
    n_time = jump_p_all_chain.shape[0]
    null_proj_l = []
    
    
    if jump_p_all_chain.ndim == 1:
        non_jump_all_chain = jump_p_all_chain<jump_p_thresh
    else:
        non_jump_all_chain = (jump_p_all_chain<jump_p_thresh).all(axis=1)
    non_jump_all_chain = nap.Tsd(d=non_jump_all_chain,t=spk_mat.t)
    consec_diff = np.zeros(len(non_jump_all_chain.d))
    consec_diff[1:]=posterior_latent_map[:-1].d!=posterior_latent_map[1:].d

    ind_all = np.arange(len(non_jump_all_chain))
    ma=np.logical_and(consec_diff, non_jump_all_chain.d.astype(bool))
    ind_to_select_from = ind_all[ma]

    sh_ind = np.random.choice(ind_to_select_from,n_shuffle,replace=False)
    

    proj_sh_peri_event_l = []
    sh_seq_l = []
    
    
    for i_sh, si in enumerate(tqdm.tqdm(sh_ind)):
        
        
        sh_seq=posterior_latent_map[si-1],posterior_latent_map[si]
        sh_seq_l.append(sh_seq)

        proj_sh,contrast_axis_sh=vlj.get_contrast_axis_and_proj(spk_mat,tuning_fit,posterior_latent_map[si-1],posterior_latent_map[si],map_state_win=contrast_axis_latent_window)
        
        sh_seq_occ_t,sh_seq_occ_ind=ah.get_sequence_occurence(sh_seq,posterior_latent_map[(posterior_latent_map.t>(proj_sh.t[0]+peri_event_win)) & (posterior_latent_map.t<(proj_sh.t[-1]-peri_event_win))],latent_distance_thresh=latent_distance_thresh)
        
        proj_sh_peri_event=nap.compute_perievent_continuous(proj_sh,sh_seq_occ_t,peri_event_win).mean(axis=1)
        
        
        proj_sh_peri_event_l.append(proj_sh_peri_event)

    proj_sh_peri_event_l=np.stack(proj_sh_peri_event_l,axis=1)
    sh_seq_l = np.array(sh_seq_l)
    
    return proj_sh_peri_event_l,sh_seq_l


def analyze_replay_unsupervised(
    model_fit,  
    prep_res,
    pbe_dt=0.02,
    data_dir_full='./',
    force_reload=False,
    dosave=True,
    save_dir=None,
    save_fn='analyze_replay_unsupervised.pkl',
    final_gain=None,
    pbe_kwargs=None,
    behavior_kwargs=None,
    gain_sweep_kwargs=None,
    decode_compare_kwargs=None,
    decode_multiple_transition_kwargs=None,
    nb_decode_kwargs=None,
    use_multi_dynamics=False,
    force_reload_multi_dynamics=False,
    skip_compare_transition=False,
    neuron_indices=None,
    verbose=True,
):
    '''
    main function to analyze replay unsupervised
    given fitted model and data
    - find pbe (can load)
    - bin spikes
    - get behavior classification
    - get transition per behavior type
    - sweep gain on shuffle test to find best gain
    - decode with transition per behavior type
    - gather unsupervised replay classification + shuffle test significance (without dynamics)

    final_gain:
    - if None (default): sweep gain (as before) and decode using best_gain
    - if not None: override gain used for shuffle-test + decoding, without recomputing base (PBE/binning/behavior/transitions)

    force_reload:
    - False: load cached if exists; if final_gain is provided and differs from cached decode gain, redo decode-stage only
    - 'decode': redo decode-stage only (shuffle-test + decoding), reuse cached base if available
    - True or 'all': recompute everything (base + sweep + decode). If final_gain is provided, decode uses final_gain but sweep is still recomputed (reported in sweep_gain_res).

    force_reload_multi_dynamics:
    - If True, force recompute multi-dynamics part even if it exists in cache. Useful when changing p_stay or other decode_multiple_transition_kwargs.

    prep_res dictionary contains:
    - spk_mat: n_time x n_neuron
    - position_tsdf: nap.TsdFrame, position in x,y
    - speed_tsd: nap.TsdFrame, speed in x,y
    - spk_times: nap.TsdFrame, spike times
    - ep_full: nap.IntervalSet, full episode intervals
    - sleep_ep: nap.IntervalSet, sleep episode intervals
    - ripple_intervals: nap.IntervalSet, ripple intervals
    '''
    def _json_default(o):
        try:
            return float(o)
        except Exception:
            return str(o)

    def _normalize_force_reload(force_reload):
        if isinstance(force_reload, str):
            fr = str(force_reload).lower()
            if fr in ['none', 'false', '0']:
                return 'none'
            if fr in ['decode']:
                return 'decode'
            if fr in ['all', 'true', '1']:
                return 'all'
            raise ValueError(f"force_reload str must be one of ['decode','all','none']; got {force_reload}")
        return 'all' if bool(force_reload) else 'none'

    if save_dir is None:
        save_dir = os.path.join(str(data_dir_full), 'py_data', 'analyze_replay_unsupervised')
    save_path = os.path.join(str(save_dir), str(save_fn))
    config_path = save_path + '.hyperparams.json'

    os.makedirs(save_dir, exist_ok=True)
    reload_mode = _normalize_force_reload(force_reload)

    pbe_kwargs_ = {
        'bin_size': 0.002,
        'smooth_std': 0.0075,
        'z_thresh': 3.0,
        'min_duration': 0.05,
        'max_duration': 0.5,
        'return_population_rate': False,
        # allow override:
        # - ep=None -> use ep_full (default behavior)
        # - threshold_ep=None -> use sleep_ep (default behavior)
        'ep': None,
        'threshold_ep': None,
        'save_dir': os.path.join(str(data_dir_full), 'py_data'),
        'save_fn': 'pbe.pkl',
        'force_reload': False,
        'dosave': True,
    }
    if pbe_kwargs is not None:
        pbe_kwargs_.update(dict(pbe_kwargs))

    behavior_kwargs_ = {
        'offmaze_method': 'roman_tmaze_projected',
        'speed_immo_thresh': 2.5,
        'speed_loco_thresh': 5.0,
    }
    if behavior_kwargs is not None:
        behavior_kwargs_.update(dict(behavior_kwargs))

    gain_sweep_kwargs_ = {
        'min_gain': 1,
        'max_gain': 10,
        'gain_step': 2,
        'train_frac': 0.5,
        'train_seed': 123,
        'n_shuffle': 100,
        'sig_thresh': 0.95,
        'q_l': None,
        'seed': 0,
        'tuning_is_count_per_bin': True,
        'decoding_kwargs': None,
        'return_full_res': False,
        'dosave': False,
        'force_reload': False,
        'save_dir': None,
        'save_fn': 'sweep_gain_shuffle_test_naive_bayes_marginal_l.pkl',
    }
    if gain_sweep_kwargs is not None:
        gain_sweep_kwargs_.update(dict(gain_sweep_kwargs))

    decode_compare_kwargs_ = {
        'n_trial_per_chunk': 400,
        'prior_magnifier': 1.0,
        'return_numpy': True,
    }
    if decode_compare_kwargs is not None:
        decode_compare_kwargs_.update(dict(decode_compare_kwargs))

    decode_multiple_transition_kwargs_ = {'p_stay': 0.95}  # default p_stay for multi-dynamics analysis
    if decode_multiple_transition_kwargs is not None:
        decode_multiple_transition_kwargs_.update(dict(decode_multiple_transition_kwargs))

    nb_decode_kwargs_ = {
        'n_time_per_chunk': 20000,
    }
    if nb_decode_kwargs is not None:
        nb_decode_kwargs_.update(dict(nb_decode_kwargs))

    hyperparams = {
        'pbe_dt': float(pbe_dt),
        'final_gain': None if final_gain is None else float(final_gain),
        'force_reload': str(reload_mode),
        'pbe_kwargs': pbe_kwargs_,
        'behavior_kwargs': behavior_kwargs_,
        'gain_sweep_kwargs': gain_sweep_kwargs_,
        'decode_compare_kwargs': decode_compare_kwargs_,
        'decode_multiple_transition_kwargs': decode_multiple_transition_kwargs_,
        'nb_decode_kwargs': nb_decode_kwargs_,
        'use_multi_dynamics': bool(use_multi_dynamics),
        'force_reload_multi_dynamics': bool(force_reload_multi_dynamics),
        'skip_compare_transition': bool(skip_compare_transition),
        'save_dir': str(save_dir),
        'save_fn': str(save_fn),
        'data_dir_full': str(data_dir_full),
    }

    if bool(verbose):
        print('[analyze_replay_unsupervised] start')
        print(f'[analyze_replay_unsupervised] save_path: {save_path}')

    t0 = time.perf_counter()

    # ---- pull light inputs (needed for decode-stage + for full recompute) ----
    spk_mat = prep_res['spk_mat']
    spk_times = prep_res['spk_times']
    model_fit_dt = float(np.median(np.diff(np.asarray(spk_mat.t))))

    tuning_fit = prep_res.get('tuning_fit', None)
    if tuning_fit is None:
        tuning_fit = model_fit.tuning

    def _compute_base():
        # ---- pull inputs ----
        position_tsdf = prep_res['position_tsdf']
        speed_tsd = prep_res['speed_tsd']
        ep_full = prep_res['full_ep']
        sleep_ep = prep_res['sleep_state_intervals_NREMepisode']
        ripple_intervals = prep_res['ripple_intervals']

        if bool(verbose):
            print(f'[analyze_replay_unsupervised] model_fit_dt={model_fit_dt:.6g}s')

        # ---- 1) PBE detection ----
        if bool(verbose):
            print('[analyze_replay_unsupervised] 1/6 detect PBE')
        spk_times_pyr = spk_times[spk_times['is_pyr']]
        ep_use = ep_full if (pbe_kwargs_.get('ep', None) is None) else pbe_kwargs_['ep']
        threshold_ep_use = sleep_ep if (pbe_kwargs_.get('threshold_ep', None) is None) else pbe_kwargs_['threshold_ep']
        pbe_res = gew.detect_population_burst_event(
            spk_times_pyr,
            mask=None,
            ep=ep_use,
            threshold_ep=threshold_ep_use,
            bin_size=float(pbe_kwargs_['bin_size']),
            smooth_std=float(pbe_kwargs_['smooth_std']),
            z_thresh=float(pbe_kwargs_['z_thresh']),
            min_duration=float(pbe_kwargs_['min_duration']),
            max_duration=float(pbe_kwargs_['max_duration']),
            ripple_intervals=ripple_intervals,
            return_population_rate=bool(pbe_kwargs_['return_population_rate']),
            save_dir=str(pbe_kwargs_['save_dir']),
            save_fn=str(pbe_kwargs_['save_fn']),
            force_reload=bool(pbe_kwargs_['force_reload']),
            dosave=bool(pbe_kwargs_['dosave']),
        )  # output

        # ---- 2) bin spikes into event tensor/mat ----
        if bool(verbose):
            print('[analyze_replay_unsupervised] 2/6 bin spikes in PBE')
        binsize = float(pbe_dt)  # output
        spk_tensor_res = tri.bin_spike_train_to_trial_based(spk_times_pyr, pbe_res['event_windows'], binsize=binsize)  # output

        # ---- 3) behavior epochs ----
        if bool(verbose):
            print('[analyze_replay_unsupervised] 3/6 behavior epochs')
        bc_res = bct.get_behavior_ep_d(
            position_tsdf,
            speed_tsd,
            offmaze_method=str(behavior_kwargs_['offmaze_method']),
            speed_immo_thresh=float(behavior_kwargs_['speed_immo_thresh']),
            speed_loco_thresh=float(behavior_kwargs_['speed_loco_thresh']),
        )
        bc_ep_d = bc_res['ep_d']  # output

        # ---- 4) transitions by behavior ----
        if bool(verbose):
            print('[analyze_replay_unsupervised] 4/6 transitions by behavior')
        decode_res = model_fit.decode_latent(spk_mat)
        posterior_latent_marg = decode_res['posterior_latent_marg']
        trans_mat_d = trans.transition_by_epoch(posterior_latent_marg, bc_ep_d)['trans_mat']  # output

        return {
            'pbe_res': pbe_res,
            'spk_tensor_res': spk_tensor_res,
            'bc_ep_d': bc_ep_d,
            'trans_mat_d': trans_mat_d,
            'model_fit_dt': float(model_fit_dt),
            'tuning_fit': tuning_fit,
            'pbe_dt': float(pbe_dt),
        }

    def _compute_multi_dynamics_only(base_res, gain_used, event_df_joint, decode_multiple_transition_kwargs=None):
        """Run only decode_with_multiple_transition_kernels and merge mean_* into event_df_joint. Independent module."""
        spk_tensor_res = base_res['spk_tensor_res']
        spk_tensor = spk_tensor_res['spike_tensor']
        tensor_pad_mask = spk_tensor_res['mask']
        time_l = spk_tensor_res['time_l']
        binsize = float(base_res['pbe_dt'])
        model_fit_dt_use = float(base_res['model_fit_dt'])
        decode_multi_kw = dict(
            tensor_pad_mask=tensor_pad_mask,
            time_l=time_l,
            dt=float(binsize),
            gain=float(gain_used),
            model_fit_dt=float(model_fit_dt_use),
            n_trial_per_chunk=int(decode_compare_kwargs_['n_trial_per_chunk']),
            prior_magnifier=float(decode_compare_kwargs_['prior_magnifier']),
            return_numpy=bool(decode_compare_kwargs_['return_numpy']),
        )
        if decode_multiple_transition_kwargs:
            decode_multi_kw.update(decode_multiple_transition_kwargs)
        pbe_multiple_transition_res = trans.decode_with_multiple_transition_kernels(
            spk_tensor, model_fit, base_res['trans_mat_d'], **decode_multi_kw)
        mean_prob = pbe_multiple_transition_res['mean_prob_category_per_event_df'].copy()
        mean_prob.columns = ['mean_' + str(c) for c in mean_prob.columns]
        event_df_joint_out = pd.concat([event_df_joint, mean_prob], axis=1)
        return pbe_multiple_transition_res, event_df_joint_out

    def _compute_decode_stage(base_res, gain_used, *, sweep_gain_res_keep=None, use_multi_dynamics=False, decode_multiple_transition_kwargs=None, skip_compare_transition=False):
        binsize = float(base_res['pbe_dt'])
        spk_tensor_res = base_res['spk_tensor_res']
        spk_tensor = spk_tensor_res['spike_tensor']
        spk_mat_pbe = spk_tensor_res['spike_mat']
        tensor_pad_mask = spk_tensor_res['mask']
        time_l = spk_tensor_res['time_l']
        event_index_per_bin = spk_tensor_res['event_index_per_bin']
        model_fit_dt_use = float(base_res['model_fit_dt'])

        # shuffle test at gain_used
        if bool(verbose):
            print(f'[analyze_replay_unsupervised] shuffle-test at gain_used={float(gain_used):.6g}')
        shuffle_test_res = shuf.shuffle_test_naive_bayes_marginal_l(
            spk_mat_pbe,
            event_index_per_bin,
            tuning=base_res['tuning_fit'],
            n_shuffle=int(gain_sweep_kwargs_['n_shuffle']),
            sig_thresh=float(gain_sweep_kwargs_['sig_thresh']),
            q_l=gain_sweep_kwargs_['q_l'],
            seed=int(gain_sweep_kwargs_['seed']),
            dt=float(binsize),
            gain=float(gain_used),
            model_fit_dt=float(model_fit_dt_use),
            tuning_is_count_per_bin=bool(gain_sweep_kwargs_['tuning_is_count_per_bin']),
            decoding_kwargs=gain_sweep_kwargs_['decoding_kwargs'],
            dosave=False,
            force_reload=False,
            save_dir=None,
            save_fn='shuffle_test_naive_bayes_marginal_l_gain_used.pkl',
            return_shuffle=False,
        )
        event_df = shuffle_test_res['event_df']

        # NB decode at gain_used (no dynamics)
        if bool(verbose):
            print('[analyze_replay_unsupervised] NB decode (naive bayes, no dynamics)')
        tuning_hz = np.asarray(model_fit.tuning) / float(model_fit_dt_use)
        nb_dt = float(binsize) * float(gain_used)
        nb_decode_res = model_fit.decode_latent_naive_bayes(
            spk_mat_pbe,
            tuning=tuning_hz,
            dt_l=nb_dt,
            n_time_per_chunk=int(nb_decode_kwargs_['n_time_per_chunk']),
        )  # output
        nb_decode_res['meta'] = {
            'model_fit_dt': float(model_fit_dt_use),
            'dt': float(binsize),
            'gain': float(gain_used),
            'dt_used_in_nb': float(nb_dt),
            'tuning_scaled_as': 'tuning_hz = model_fit.tuning / model_fit_dt; dt_l = dt * gain',
        }

        # decode PBE under behavior-specific transitions (skip if skip_compare_transition and using multi-dynamics)
        pbe_compare_transition_res = None
        if skip_compare_transition:
            event_df_joint = event_df.copy()
        else:
            if bool(verbose):
                print('[analyze_replay_unsupervised] decode PBE under behavior-specific transitions')
            pbe_compare_transition_res = trans.decode_compare_transition_kernels(
                spk_tensor,
                model_fit,
                base_res['trans_mat_d'],
                tensor_pad_mask=tensor_pad_mask,
                time_l=time_l,
                dt=float(binsize),
                gain=float(gain_used),
                model_fit_dt=float(model_fit_dt_use),
                n_trial_per_chunk=int(decode_compare_kwargs_['n_trial_per_chunk']),
                prior_magnifier=float(decode_compare_kwargs_['prior_magnifier']),
                return_numpy=bool(decode_compare_kwargs_['return_numpy']),
            )  # output
            prob_category_per_event_df = pbe_compare_transition_res['prob_category_per_event_df']
            event_df_joint = pd.concat([event_df, prob_category_per_event_df], axis=1)  # output

        pbe_multiple_transition_res = None
        if use_multi_dynamics:
            if bool(verbose):
                print('[analyze_replay_unsupervised] decode PBE with multiple transition kernels (multi-dynamics)')
            pbe_multiple_transition_res, event_df_joint = _compute_multi_dynamics_only(
                base_res, gain_used, event_df_joint, decode_multiple_transition_kwargs=decode_multiple_transition_kwargs)

        out = {
            'sweep_gain_res': sweep_gain_res_keep,
            'shuffle_test_res_gain_used': shuffle_test_res,
            'best_gain': float(gain_used),
            'nb_decode_res': nb_decode_res,
            'pbe_compare_transition_res': pbe_compare_transition_res,
            'pbe_multiple_transition_res': pbe_multiple_transition_res,
            'event_df_joint': event_df_joint,
        }
        return out

    # ---- load cached / decide what to recompute ----
    cached = None
    if os.path.exists(save_path):
        cached = pd.read_pickle(save_path)
        if bool(verbose):
            print(f'[analyze_replay_unsupervised] exists: {save_path}')

    if cached is not None and reload_mode in ['none', 'decode']:
        hp = cached.get('hyperparams', {})
        cached_use_multi = hp.get('use_multi_dynamics', False)
        cached_skip_compare = hp.get('skip_compare_transition', False)
        if reload_mode == 'none' and final_gain is None and neuron_indices is None and cached_use_multi == use_multi_dynamics and cached_skip_compare == skip_compare_transition and not force_reload_multi_dynamics:
            print(f'[analyze_replay_unsupervised] loading cached (force_reload=False, final_gain=None, use_multi_dynamics={use_multi_dynamics}, skip_compare_transition={skip_compare_transition}, force_reload_multi_dynamics=False): {save_path}')
            return cached
        cached_pbe_dt = hp.get('pbe_dt', None)
        if cached_pbe_dt is not None and float(cached_pbe_dt) != float(pbe_dt):
            if reload_mode == 'decode' or final_gain is not None:
                raise ValueError(f"decode-only reuse requires pbe_dt match cached (cached={cached_pbe_dt}, requested={pbe_dt}). Use force_reload='all'.")

    def _subselect_spk_tensor_res_neuron(spk_tensor_res, neuron_indices):
        """Subselect neuron dimension; indices are into the full (e.g. pyr) population."""
        idx = np.asarray(neuron_indices, dtype=int)
        out = dict(spk_tensor_res)
        out['spike_tensor'] = np.asarray(spk_tensor_res['spike_tensor'])[..., idx]
        sm = spk_tensor_res['spike_mat']
        if hasattr(sm, 'd'):
            out['spike_mat'] = nap.TsdFrame(t=sm.t, d=np.asarray(sm.d)[:, idx])
        else:
            out['spike_mat'] = np.asarray(sm)[:, idx]
        return out

    # ---- compute base (either reuse cache base or recompute all) ----
    if cached is not None and reload_mode in ['none', 'decode']:
        base_res = {
            'pbe_res': cached['pbe_res'],
            'spk_tensor_res': cached['spk_tensor_res'],
            'bc_ep_d': cached['bc_ep_d'],
            'trans_mat_d': cached['trans_mat_d'],
            'model_fit_dt': float(cached.get('nb_decode_res', {}).get('meta', {}).get('model_fit_dt', model_fit_dt)),
            'tuning_fit': tuning_fit,
            'pbe_dt': float(cached.get('hyperparams', {}).get('pbe_dt', pbe_dt)),
        }
        if neuron_indices is not None:
            idx = np.asarray(neuron_indices, dtype=int)
            base_res['spk_tensor_res'] = _subselect_spk_tensor_res_neuron(base_res['spk_tensor_res'], idx)
            # tuning_fit already from prep_res / model_fit (subsampled); leave base_res['tuning_fit'] as is
        sweep_gain_res_keep = cached.get('sweep_gain_res', None)
        cached_gain = cached.get('best_gain', None)
        if final_gain is None:
            if cached_gain is None:
                raise ValueError("decode-only reuse needs cached['best_gain'] when final_gain is None. Use force_reload='all' or provide final_gain.")
            gain_used = float(cached_gain)
        else:
            gain_used = float(final_gain)

        if reload_mode == 'none' and (final_gain is not None) and (cached_gain is not None) and (float(cached_gain) == float(gain_used)) and (cached_use_multi == use_multi_dynamics) and (cached_skip_compare == skip_compare_transition) and not force_reload_multi_dynamics:
            if bool(verbose):
                print(f'[analyze_replay_unsupervised] cached decode already uses gain={float(gain_used):.6g}; returning cached')
            return cached

        # When only force_reload_multi_dynamics: update whole return from cache, recompute just multi-dynamics and overwrite those keys
        if force_reload_multi_dynamics and use_multi_dynamics and cached_use_multi == use_multi_dynamics:
            if bool(verbose):
                print(f'[analyze_replay_unsupervised] force_reload_multi_dynamics=True: recomputing multi-dynamics only, updating full return')
            out = dict(cached)
            event_df_base = out['event_df_joint'].copy()
            mean_cols = [c for c in event_df_base.columns if str(c).startswith('mean_')]
            if mean_cols:
                event_df_base = event_df_base.drop(columns=mean_cols)
            pbe_multi, event_df_joint_new = _compute_multi_dynamics_only(
                base_res, gain_used, event_df_base, decode_multiple_transition_kwargs=decode_multiple_transition_kwargs_)
            out['pbe_multiple_transition_res'] = pbe_multi
            out['event_df_joint'] = event_df_joint_new
            out['best_gain'] = float(gain_used)
            out_h = dict(out.get('hyperparams', {}))
            out_h.update({
                'final_gain': None if final_gain is None else float(final_gain),
                'force_reload': str(reload_mode),
                'gain_used': float(gain_used),
                'use_multi_dynamics': bool(use_multi_dynamics),
                'force_reload_multi_dynamics': bool(force_reload_multi_dynamics),
            })
            out['hyperparams'] = out_h
            if bool(dosave):
                pd.to_pickle(out, save_path)
                print(f'[analyze_replay_unsupervised] saved: {save_path}')
                with open(config_path, 'w') as f:
                    json.dump(out_h, f, indent=2, sort_keys=True, default=_json_default)
                print(f'[analyze_replay_unsupervised] saved: {config_path}')
            return out

        if cached_use_multi == use_multi_dynamics:
            decode_stage = _compute_decode_stage(base_res, gain_used, sweep_gain_res_keep=sweep_gain_res_keep,
                                                use_multi_dynamics=use_multi_dynamics,
                                                decode_multiple_transition_kwargs=decode_multiple_transition_kwargs_,
                                                skip_compare_transition=skip_compare_transition)
        else:
            if bool(verbose):
                print(f'[analyze_replay_unsupervised] add/strip multi-dynamics only (reload_mode={reload_mode}) gain_used={float(gain_used):.6g}')
            decode_stage = {
                'sweep_gain_res': sweep_gain_res_keep,
                'shuffle_test_res_gain_used': cached.get('shuffle_test_res_gain_used'),
                'best_gain': float(gain_used),
                'nb_decode_res': cached['nb_decode_res'],
                'pbe_compare_transition_res': cached['pbe_compare_transition_res'],
                'pbe_multiple_transition_res': None,
                'event_df_joint': cached['event_df_joint'].copy(),
            }
            if use_multi_dynamics:
                if force_reload_multi_dynamics:
                    if bool(verbose):
                        print(f'[analyze_replay_unsupervised] force_reload_multi_dynamics=True: recomputing multi-dynamics')
                    pbe_multi, event_df_joint_new = _compute_multi_dynamics_only(
                        base_res, gain_used, decode_stage['event_df_joint'], decode_multiple_transition_kwargs=decode_multiple_transition_kwargs_)
                    decode_stage['pbe_multiple_transition_res'] = pbe_multi
                    decode_stage['event_df_joint'] = event_df_joint_new
                else:
                    cached_multi = cached.get('pbe_multiple_transition_res')
                    if cached_multi is not None:
                        mean_prob = cached_multi['mean_prob_category_per_event_df'].copy()
                        mean_prob.columns = ['mean_' + str(c) for c in mean_prob.columns]
                        decode_stage['event_df_joint'] = pd.concat([decode_stage['event_df_joint'], mean_prob], axis=1)
                        decode_stage['pbe_multiple_transition_res'] = cached_multi
                    else:
                        pbe_multi, event_df_joint_new = _compute_multi_dynamics_only(
                            base_res, gain_used, decode_stage['event_df_joint'], decode_multiple_transition_kwargs=decode_multiple_transition_kwargs_)
                        decode_stage['pbe_multiple_transition_res'] = pbe_multi
                        decode_stage['event_df_joint'] = event_df_joint_new
            else:
                mean_cols = [c for c in decode_stage['event_df_joint'].columns if str(c).startswith('mean_')]
                if mean_cols:
                    decode_stage['event_df_joint'] = decode_stage['event_df_joint'].drop(columns=mean_cols)

        out = dict(cached)
        out.update(decode_stage)
        out['best_gain'] = float(gain_used)
        out_h = dict(out.get('hyperparams', {}))
        out_h.update({
            'final_gain': None if final_gain is None else float(final_gain),
            'force_reload': str(reload_mode),
            'gain_used': float(gain_used),
            'use_multi_dynamics': bool(use_multi_dynamics),
            'force_reload_multi_dynamics': bool(force_reload_multi_dynamics),
        })
        out['hyperparams'] = out_h

        if bool(dosave):
            pd.to_pickle(out, save_path)
            print(f'[analyze_replay_unsupervised] saved: {save_path}')
            with open(config_path, 'w') as f:
                json.dump(out_h, f, indent=2, sort_keys=True, default=_json_default)
            print(f'[analyze_replay_unsupervised] saved: {config_path}')
        return out

    # full recompute (reload_mode == 'all' or no cache)
    base_res = _compute_base()

    # ---- 5) gain sweep for shuffle test ----
    # In force_reload='all' mode, we recompute sweep even if final_gain is provided
    # (final_gain still overrides gain used for decode-stage).
    do_sweep = (final_gain is None) or (reload_mode == 'all')
    sweep_keep = None
    if bool(do_sweep):
        if bool(verbose):
            if final_gain is None:
                print('[analyze_replay_unsupervised] 5/6 sweep gain (shuffle test)')
            else:
                print('[analyze_replay_unsupervised] 5/6 sweep gain (shuffle test; final_gain override enabled)')
        spk_tensor_res = base_res['spk_tensor_res']
        sweep_gain_res = shuf.sweep_gain_shuffle_test_naive_bayes_marginal_l(
            spk_tensor_res['spike_mat'],
            spk_tensor_res['event_index_per_bin'],
            tuning=base_res['tuning_fit'],
            min_gain=float(gain_sweep_kwargs_['min_gain']),
            max_gain=float(gain_sweep_kwargs_['max_gain']),
            gain_step=float(gain_sweep_kwargs_['gain_step']),
            train_frac=float(gain_sweep_kwargs_['train_frac']),
            train_seed=int(gain_sweep_kwargs_['train_seed']),
            n_shuffle=int(gain_sweep_kwargs_['n_shuffle']),
            sig_thresh=float(gain_sweep_kwargs_['sig_thresh']),
            q_l=gain_sweep_kwargs_['q_l'],
            seed=int(gain_sweep_kwargs_['seed']),
            dt=float(base_res['pbe_dt']),
            model_fit_dt=float(base_res['model_fit_dt']),
            tuning_is_count_per_bin=bool(gain_sweep_kwargs_['tuning_is_count_per_bin']),
            decoding_kwargs=gain_sweep_kwargs_['decoding_kwargs'],
            dosave=bool(gain_sweep_kwargs_['dosave']),
            force_reload=bool(gain_sweep_kwargs_['force_reload']),
            save_dir=gain_sweep_kwargs_['save_dir'],
            save_fn=str(gain_sweep_kwargs_['save_fn']),
            return_full_res=bool(gain_sweep_kwargs_['return_full_res']),
        )  # output
        sweep_keep = sweep_gain_res
        if bool(verbose):
            print(f'[analyze_replay_unsupervised] sweep best_gain={float(sweep_gain_res["best_gain"]):.6g}')

    if final_gain is None:
        if sweep_keep is None:
            raise ValueError('internal error: expected sweep_keep when final_gain is None')
        gain_used = float(sweep_keep['best_gain'])
        if bool(verbose):
            print(f'[analyze_replay_unsupervised] gain_used(best_gain)={gain_used:.6g}')
    else:
        gain_used = float(final_gain)
        if bool(verbose):
            print(f'[analyze_replay_unsupervised] gain_used(final_gain)={gain_used:.6g}')

    decode_stage = _compute_decode_stage(base_res, gain_used, sweep_gain_res_keep=sweep_keep,
                                        use_multi_dynamics=use_multi_dynamics,
                                        decode_multiple_transition_kwargs=decode_multiple_transition_kwargs_,
                                        skip_compare_transition=skip_compare_transition)

    unsup_res = {
        'hyperparams': hyperparams,
        'pbe_res': base_res['pbe_res'],  # output
        'spk_tensor_res': base_res['spk_tensor_res'],  # output
        'bc_ep_d': base_res['bc_ep_d'],  # output
        'trans_mat_d': base_res['trans_mat_d'],  # output
        'sweep_gain_res': decode_stage.get('sweep_gain_res', None),  # output
        'best_gain': float(gain_used),  # output (gain used in decode-stage)
        'nb_decode_res': decode_stage['nb_decode_res'],  # output
        'pbe_compare_transition_res': decode_stage['pbe_compare_transition_res'],  # output
        'pbe_multiple_transition_res': decode_stage.get('pbe_multiple_transition_res'),  # output
        'event_df_joint': decode_stage['event_df_joint'],  # output
        'shuffle_test_res_gain_used': decode_stage.get('shuffle_test_res_gain_used', None),  # output
    }
    hyperparams2 = dict(hyperparams)
    hyperparams2.update({'gain_used': float(gain_used)})
    unsup_res['hyperparams'] = hyperparams2

    if bool(dosave):
        pd.to_pickle(unsup_res, save_path)
        print(f'[analyze_replay_unsupervised] saved: {save_path}')
        with open(config_path, 'w') as f:
            json.dump(hyperparams2, f, indent=2, sort_keys=True, default=_json_default)
        print(f'[analyze_replay_unsupervised] saved: {config_path}')

    if bool(verbose):
        dt_s = time.perf_counter() - t0
        print(f'[analyze_replay_unsupervised] done in {dt_s:.2f}s; n_event={len(unsup_res["event_df_joint"])}')
    return unsup_res
    



def analyze_replay_supervised(
    prep_res,
    data_dir_full='./',
    force_reload=False,
    dosave=True,
    save_dir=None,
    save_fn='analyze_replay_supervised.pkl',
    pbe_dt = 0.02,
    final_gain=None,
    pbe_kwargs=None,
    tuning_kwargs=None,
    decode_kwargs=None,
    gain_sweep_kwargs=None,
    verbose=True,
):
    '''
    main function to analyze replay supervised
    given fitted model and data
    - find pbe (can load)
    - bin spikes
    - get tuning
    - sweep gain on shuffle test to find best gain
    - decode with dynamics
    - gather supervised replay metrics + shuffle test significance (without dynamics)

    final_gain:
    - if None (default): sweep gain (as before) and decode using best_gain
    - if not None: override gain used for shuffle-test + decoding + replay metrics, without recomputing base (tuning/PBE/binning)

    force_reload:
    - False: load cached if exists; if final_gain is provided and differs from cached decode gain, redo decode-stage only
    - 'decode': redo decode-stage only (shuffle-test + decoding + replay metrics), reuse cached base if available
    - True or 'all': recompute everything (base + sweep + decode). If final_gain is provided, decode uses final_gain but sweep is still recomputed (reported in sweep_gain_res).
    '''

    def _json_default(o):
        try:
            return float(o)
        except Exception:
            return str(o)

    def _normalize_force_reload(force_reload):
        if isinstance(force_reload, str):
            fr = str(force_reload).lower()
            if fr in ['none', 'false', '0']:
                return 'none'
            if fr in ['decode']:
                return 'decode'
            if fr in ['all', 'true', '1']:
                return 'all'
            raise ValueError(f"force_reload str must be one of ['decode','all','none']; got {force_reload}")
        return 'all' if bool(force_reload) else 'none'

    if save_dir is None:
        save_dir = os.path.join(str(data_dir_full), 'py_data', 'analyze_replay_supervised')
    save_path = os.path.join(str(save_dir), str(save_fn))
    config_path = save_path + '.hyperparams.json'

    os.makedirs(save_dir, exist_ok=True)
    reload_mode = _normalize_force_reload(force_reload)

    pbe_kwargs_ = {
        'bin_size': 0.002,
        'smooth_std': 0.0075,
        'z_thresh': 3.0,
        'min_duration': 0.05,
        'max_duration': 0.5,
        'return_population_rate': False,
        # allow override:
        # - ep=None -> use ep_full (default behavior)
        # - threshold_ep=None -> use sleep_ep (default behavior)
        'ep': None,
        'threshold_ep': None,
        'save_dir': os.path.join(str(data_dir_full), 'py_data'),
        'save_fn': 'pbe.pkl',
        'force_reload': False,
        'dosave': True,
    }
    if pbe_kwargs is not None:
        pbe_kwargs_.update(dict(pbe_kwargs))

    tuning_kwargs_ = {
        'label_bin_size': 3.0,
        'smooth_std': 3.0,
        'occupancy_threshold': 0.1,
        'speed_thresh': 5.0,
        'spk_count_bin_size_for_tuning': 0.1,
    }
    if tuning_kwargs is not None:
        tuning_kwargs_.update(dict(tuning_kwargs))

    gain_sweep_kwargs_ = {
        'min_gain': 1,
        'max_gain': 20,
        'gain_step': 2,
        'train_frac': 0.5,
        'train_seed': 123,
        'n_shuffle': 100,
        'sig_thresh': 0.95,
        'q_l': None,
        'seed': 0,
        'decoding_kwargs': None,
        'return_full_res': False,
    }
    if gain_sweep_kwargs is not None:
        gain_sweep_kwargs_.update(dict(gain_sweep_kwargs))

    decode_kwargs_ = {
        'continuous_transition_movement_variance': 10.,
        'p_move_to_jump': 0.02,
        'p_jump_to_move': 0.02,
        'n_time_per_chunk': 20000,
        'likelihood_scale': 1.0,
        'observation_model': 'poisson',
        'prior_magnifier': 1.0,
        'n_trial_per_chunk': 200,
    }
    if decode_kwargs is not None:
        decode_kwargs_.update(dict(decode_kwargs))

    hyperparams = {
        'pbe_dt': float(pbe_dt),
        'final_gain': None if final_gain is None else float(final_gain),
        'force_reload': str(reload_mode),
        'pbe_kwargs': pbe_kwargs_,
        'tuning_kwargs': tuning_kwargs_,
        'gain_sweep_kwargs': gain_sweep_kwargs_,
        'decode_kwargs': decode_kwargs_,
        'save_dir': str(save_dir),
        'save_fn': str(save_fn),
        'data_dir_full': str(data_dir_full),
    }

    if bool(verbose):
        print('[analyze_replay_supervised] start')
        print(f'[analyze_replay_supervised] save_path: {save_path}')

    t0 = time.perf_counter()

    # ---- load cached / decide what to recompute ----
    cached = None
    if os.path.exists(save_path):
        cached = pd.read_pickle(save_path)
        if bool(verbose):
            print(f'[analyze_replay_supervised] exists: {save_path}')

    if cached is not None and reload_mode == 'none' and final_gain is None:
        print(f'[analyze_replay_supervised] loading cached (force_reload=False, final_gain=None): {save_path}')
        return cached

    if cached is not None and reload_mode in ['none', 'decode']:
        hp = cached.get('hyperparams', {})
        cached_pbe_dt = hp.get('pbe_dt', None)
        if cached_pbe_dt is not None and float(cached_pbe_dt) != float(pbe_dt):
            if reload_mode == 'decode' or final_gain is not None:
                raise ValueError(f"decode-only reuse requires pbe_dt match cached (cached={cached_pbe_dt}, requested={pbe_dt}). Use force_reload='all'.")

    def _compute_base():
        # ---- pull inputs ----
        spk_times = prep_res['spk_times']
        position_tsdf = prep_res['position_tsdf']
        speed_tsd = prep_res['speed_tsd']
        behavior_ep = prep_res['behavior_ep']
        ep_full = prep_res['full_ep']
        sleep_ep = prep_res['sleep_state_intervals_NREMepisode']
        ripple_intervals = prep_res['ripple_intervals']

        # use pyr units consistently (PBE, decoding, tuning)
        spk_times_pyr = spk_times[spk_times['is_pyr']]

        # ---- 1) tuning (supervised) ----
        if bool(verbose):
            print('[analyze_replay_supervised] 1/6 tuning')
        try:
            behavior_ep0 = behavior_ep[0]
        except Exception:
            behavior_ep0 = behavior_ep

        spk_mat_for_tuning = spk_times_pyr.count(float(tuning_kwargs_['spk_count_bin_size_for_tuning']), ep=behavior_ep0)
        label_d = {
            'familiar': position_tsdf[['x', 'y']].restrict(behavior_ep0),
        }
        ep_d = {
            'familiar': speed_tsd.restrict(behavior_ep0).threshold(float(tuning_kwargs_['speed_thresh'])).time_support,
        }
        label_bin_size_d = {
            'familiar': float(tuning_kwargs_['label_bin_size']),
        }
        smooth_std_d = {
            'familiar': float(tuning_kwargs_['smooth_std']),
        }
        import poor_man_gplvm.supervised_analysis.get_tuning_supervised as gts
        tuning_res = gts.get_tuning(
            label_l=label_d,
            spk_mat=spk_mat_for_tuning,
            ep=ep_d,
            label_bin_size=label_bin_size_d,
            smooth_std=smooth_std_d,
            occupancy_threshold=float(tuning_kwargs_['occupancy_threshold']),
        )  # output

        # ---- 2) PBE detection (cache-aware) ----
        if bool(verbose):
            print('[analyze_replay_supervised] 2/6 detect PBE')
        ep_use = ep_full if (pbe_kwargs_.get('ep', None) is None) else pbe_kwargs_['ep']
        threshold_ep_use = sleep_ep if (pbe_kwargs_.get('threshold_ep', None) is None) else pbe_kwargs_['threshold_ep']
        pbe_res = gew.detect_population_burst_event(
            spk_times_pyr,
            mask=None,
            ep=ep_use,
            threshold_ep=threshold_ep_use,
            bin_size=float(pbe_kwargs_['bin_size']),
            smooth_std=float(pbe_kwargs_['smooth_std']),
            z_thresh=float(pbe_kwargs_['z_thresh']),
            min_duration=float(pbe_kwargs_['min_duration']),
            max_duration=float(pbe_kwargs_['max_duration']),
            ripple_intervals=ripple_intervals,
            return_population_rate=bool(pbe_kwargs_['return_population_rate']),
            save_dir=str(pbe_kwargs_['save_dir']),
            save_fn=str(pbe_kwargs_['save_fn']),
            force_reload=bool(pbe_kwargs_['force_reload']),
            dosave=bool(pbe_kwargs_['dosave']),
        )  # output

        # ---- 3) bin spikes into event tensor/mat ----
        if bool(verbose):
            print('[analyze_replay_supervised] 3/6 bin spikes in PBE')
        spk_tensor_res = tri.bin_spike_train_to_trial_based(
            spk_times_pyr,
            pbe_res['event_windows'],
            binsize=float(pbe_dt),
        )  # output

        return {
            'tuning_res': tuning_res,
            'pbe_res': pbe_res,
            'spk_tensor_res': spk_tensor_res,
            'pbe_dt': float(pbe_dt),
        }

    def _compute_decode_stage(base_res, gain_used, *, sweep_gain_res_keep=None):
        spk_tensor_res = base_res['spk_tensor_res']
        spk_tensor = spk_tensor_res['spike_tensor']
        tensor_pad_mask = spk_tensor_res['mask']
        time_l = spk_tensor_res['time_l']
        event_index_per_bin = spk_tensor_res['event_index_per_bin']
        spk_mat_pbe = spk_tensor_res['spike_mat']

        # shuffle test at gain_used
        decoding_kwargs_nb = {} if gain_sweep_kwargs_['decoding_kwargs'] is None else dict(gain_sweep_kwargs_['decoding_kwargs'])
        if 'n_time_per_chunk' not in decoding_kwargs_nb:
            decoding_kwargs_nb['n_time_per_chunk'] = int(decode_kwargs_['n_time_per_chunk'])
        if bool(verbose):
            print(f'[analyze_replay_supervised] shuffle-test at gain_used={float(gain_used):.6g}')
        shuffle_test_res = shuf.shuffle_test_naive_bayes_marginal_l(
            spk_mat_pbe,
            event_index_per_bin,
            tuning=base_res['tuning_res']['tuning_flat'],
            n_shuffle=int(gain_sweep_kwargs_['n_shuffle']),
            sig_thresh=float(gain_sweep_kwargs_['sig_thresh']),
            q_l=gain_sweep_kwargs_['q_l'],
            seed=int(gain_sweep_kwargs_['seed']),
            dt=float(base_res['pbe_dt']),
            gain=float(gain_used),
            model_fit_dt=1.0,
            tuning_is_count_per_bin=False,
            decoding_kwargs=decoding_kwargs_nb,
            dosave=False,
            force_reload=False,
            save_dir=None,
            save_fn='shuffle_test_supervised_nb_gain_used.pkl',
            return_shuffle=False,
        )
        event_df_shuffle = shuffle_test_res['event_df']

        # decode with dynamics
        if bool(verbose):
            print('[analyze_replay_supervised] decode with dynamics')
        import poor_man_gplvm.supervised_analysis.decoder_supervised as decoder_supervised
        decode_res = decoder_supervised.decode_with_dynamics(
            spk_tensor,
            base_res['tuning_res']['tuning_flat'],
            tensor_pad_mask=tensor_pad_mask,
            coord_to_flat_idx=base_res['tuning_res']['coord_to_flat_idx'],
            flat_idx_to_coord=base_res['tuning_res'].get('flat_idx_to_coord', None),
            dt=float(base_res['pbe_dt']),
            gain=float(gain_used),
            time_l=time_l,
            continuous_transition_movement_variance=float(decode_kwargs_['continuous_transition_movement_variance']),
            p_move_to_jump=float(decode_kwargs_['p_move_to_jump']),
            p_jump_to_move=float(decode_kwargs_['p_jump_to_move']),
            n_time_per_chunk=int(decode_kwargs_['n_time_per_chunk']),
            likelihood_scale=float(decode_kwargs_['likelihood_scale']),
            observation_model=str(decode_kwargs_['observation_model']),
            prior_magnifier=float(decode_kwargs_['prior_magnifier']),
            n_trial_per_chunk=int(decode_kwargs_['n_trial_per_chunk']),
        )  # output

        # replay metrics + join to event table
        if bool(verbose):
            print('[analyze_replay_supervised] replay metrics')
        import poor_man_gplvm.supervised_analysis.replay_metrics_supervised as rms
        stepsize_split_thresh = 10.0
        continuous_prob_thresh = 0.8
        metrics_res = rms.compute_replay_metrics(
            decode_res['posterior_latent_marg'],
            decode_res['posterior_dynamics_marg'],
            start_index=decode_res['start_index'],
            end_index=decode_res['end_index'],
            binsize=float(base_res['pbe_dt']),
            position_key={'familiar': ('x', 'y')},
            min_segment_duration=0.06,
            continuous_prob_thresh=float(continuous_prob_thresh),
            stepsize_split_thresh=float(stepsize_split_thresh),
        )  # output
        metrics_df = metrics_res['metrics_df']

        event_windows_pd = base_res['pbe_res']['event_windows'].as_dataframe().reset_index(drop=True)  # output
        metrics_df = metrics_df.reset_index(drop=True)
        event_df_shuffle = event_df_shuffle.reset_index(drop=True)
        replay_metrics_df = pd.concat([event_windows_pd, metrics_df, event_df_shuffle], axis=1)  # output

        return {
            'sweep_gain_res': sweep_gain_res_keep,
            'shuffle_test_res_gain_used': shuffle_test_res,
            'best_gain': float(gain_used),
            'decode_res': decode_res,
            'metrics_res': metrics_res,
            'replay_metrics_df': replay_metrics_df,
        }

    if cached is not None and reload_mode in ['none', 'decode']:
        base_res = {
            'tuning_res': cached['tuning_res'],
            'pbe_res': cached['pbe_res'],
            'spk_tensor_res': cached['spk_tensor_res'],
            'pbe_dt': float(cached.get('hyperparams', {}).get('pbe_dt', pbe_dt)),
        }
        sweep_gain_res_keep = cached.get('sweep_gain_res', None)
        cached_gain = cached.get('best_gain', None)
        if final_gain is None:
            if cached_gain is None:
                raise ValueError("decode-only reuse needs cached['best_gain'] when final_gain is None. Use force_reload='all' or provide final_gain.")
            gain_used = float(cached_gain)
        else:
            gain_used = float(final_gain)

        if reload_mode == 'none' and (final_gain is not None) and (cached_gain is not None) and (float(cached_gain) == float(gain_used)):
            if bool(verbose):
                print(f'[analyze_replay_supervised] cached decode already uses gain={float(gain_used):.6g}; returning cached')
            return cached

        if bool(verbose):
            print(f'[analyze_replay_supervised] redo decode-stage only (reload_mode={reload_mode}) gain_used={float(gain_used):.6g}')
        decode_stage = _compute_decode_stage(base_res, gain_used, sweep_gain_res_keep=sweep_gain_res_keep)

        out = dict(cached)
        out.update(decode_stage)
        out['best_gain'] = float(gain_used)
        out_h = dict(out.get('hyperparams', {}))
        out_h.update({
            'final_gain': None if final_gain is None else float(final_gain),
            'force_reload': str(reload_mode),
            'gain_used': float(gain_used),
        })
        out['hyperparams'] = out_h

        if bool(dosave):
            pd.to_pickle(out, save_path)
            print(f'[analyze_replay_supervised] saved: {save_path}')
            with open(config_path, 'w') as f:
                json.dump(out_h, f, indent=2, sort_keys=True, default=_json_default)
            print(f'[analyze_replay_supervised] saved: {config_path}')
        return out

    # full recompute (reload_mode == 'all' or no cache)
    base_res = _compute_base()

    # ---- gain sweep ----
    # In force_reload='all' mode, we recompute sweep even if final_gain is provided
    # (final_gain still overrides gain used for decode-stage).
    do_sweep = (final_gain is None) or (reload_mode == 'all')
    sweep_keep = None
    if bool(do_sweep):
        if bool(verbose):
            if final_gain is None:
                print('[analyze_replay_supervised] 4/6 sweep gain (shuffle test)')
            else:
                print('[analyze_replay_supervised] 4/6 sweep gain (shuffle test; final_gain override enabled)')
        decoding_kwargs_nb = {} if gain_sweep_kwargs_['decoding_kwargs'] is None else dict(gain_sweep_kwargs_['decoding_kwargs'])
        if 'n_time_per_chunk' not in decoding_kwargs_nb:
            decoding_kwargs_nb['n_time_per_chunk'] = int(decode_kwargs_['n_time_per_chunk'])
        spk_tensor_res = base_res['spk_tensor_res']
        sweep_gain_res = shuf.sweep_gain_shuffle_test_naive_bayes_marginal_l(
            spk_tensor_res['spike_mat'],
            spk_tensor_res['event_index_per_bin'],
            tuning=base_res['tuning_res']['tuning_flat'],
            min_gain=float(gain_sweep_kwargs_['min_gain']),
            max_gain=float(gain_sweep_kwargs_['max_gain']),
            gain_step=float(gain_sweep_kwargs_['gain_step']),
            train_frac=float(gain_sweep_kwargs_['train_frac']),
            train_seed=int(gain_sweep_kwargs_['train_seed']),
            n_shuffle=int(gain_sweep_kwargs_['n_shuffle']),
            sig_thresh=float(gain_sweep_kwargs_['sig_thresh']),
            q_l=gain_sweep_kwargs_['q_l'],
            seed=int(gain_sweep_kwargs_['seed']),
            dt=float(base_res['pbe_dt']),
            model_fit_dt=1.0,
            tuning_is_count_per_bin=False,
            decoding_kwargs=decoding_kwargs_nb,
            dosave=False,
            force_reload=False,
            save_dir=None,
            save_fn='sweep_gain_shuffle_test_supervised_nb.pkl',
            return_full_res=bool(gain_sweep_kwargs_['return_full_res']),
        )  # output
        sweep_keep = sweep_gain_res
        if bool(verbose):
            print(f'[analyze_replay_supervised] sweep best_gain={float(sweep_gain_res["best_gain"]):.6g}')

    if final_gain is None:
        if sweep_keep is None:
            raise ValueError('internal error: expected sweep_keep when final_gain is None')
        gain_used = float(sweep_keep['best_gain'])
        if bool(verbose):
            print(f'[analyze_replay_supervised] gain_used(best_gain)={gain_used:.6g}')
    else:
        gain_used = float(final_gain)
        if bool(verbose):
            print(f'[analyze_replay_supervised] gain_used(final_gain)={gain_used:.6g}')

    decode_stage = _compute_decode_stage(base_res, gain_used, sweep_gain_res_keep=sweep_keep)

    sup_res = {
        'hyperparams': hyperparams,
        'pbe_res': base_res['pbe_res'],
        'spk_tensor_res': base_res['spk_tensor_res'],
        'tuning_res': base_res['tuning_res'],
        'sweep_gain_res': decode_stage.get('sweep_gain_res', None),
        'best_gain': float(gain_used),
        'decode_res': decode_stage['decode_res'],
        'metrics_res': decode_stage['metrics_res'],
        'replay_metrics_df': decode_stage['replay_metrics_df'],
        'shuffle_test_res_gain_used': decode_stage.get('shuffle_test_res_gain_used', None),
    }
    hyperparams2 = dict(hyperparams)
    hyperparams2.update({'gain_used': float(gain_used)})
    sup_res['hyperparams'] = hyperparams2

    if bool(dosave):
        pd.to_pickle(sup_res, save_path)
        print(f'[analyze_replay_supervised] saved: {save_path}')
        with open(config_path, 'w') as f:
            json.dump(hyperparams2, f, indent=2, sort_keys=True, default=_json_default)
        print(f'[analyze_replay_supervised] saved: {config_path}')

    if bool(verbose):
        dt_s = time.perf_counter() - t0
        print(f'[analyze_replay_supervised] done in {dt_s:.2f}s; n_event={len(sup_res["replay_metrics_df"])}')
    return sup_res