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
import poor_man_gplvm.plot_helper as ph
import tqdm
import poor_man_gplvm.analysis_helper as ah
import poor_man_gplvm.verify_latent_jump as vlj

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
                                hide_box=True,
                                seperate_colorbar=True,
                                colorbar=False,
                                background_mode='line',
                                color_time=True,
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
        plot_maze_background(position_tsdf,ds=ds,fig=fig,ax=ax,c=maze_c,alpha=maze_alpha,mode=background_mode)


    # plot running and immobility with different marker shape
    speed_category = pd.cut(behavior_tsdf['speed_gauss'],bins=[0,*speed_category_thresh,np.inf],labels=False)
    speed_category_unique = np.unique(speed_category)
    speed_category_unique = speed_category_unique[np.logical_not(np.isnan(speed_category_unique))].astype(int)
    
    
    # Plot running and immobility
    latent_l_ind = np.arange(len(latent_l)) 
    norm=Normalize(vmin=0,vmax=len(latent_l))
    colors = cmap(norm(latent_l_ind)) # color based on the index of latent within latent_l, not the latent value
    
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
        
def plot_multiple_latent_spatial_map(latent_ind_l,posterior_latent_map,behavior_tsdf,position_tsdf=None,speed_thresh=5,color_time=True,kwargs_scatter = dict(s=10,alpha=0.5)):
    nplots = len(latent_ind_l)
    fig,axs=ph.subplots_wrapper(nplots,)
    if position_tsdf is None:
        position_tsdf = behavior_tsdf[['x','y']]
    for ii,i in enumerate(latent_ind_l):
        ax=axs.ravel()[ii]
        # state_l = np.arange(10)
        latent_l =[i]
        to_return=plot_latent_list_vs_position(latent_l, posterior_latent_map,behavior_tsdf,pos_col=['x','y'],fig=fig,ax=ax,
                                        speed_col='speed_gauss',
                                        speed_category_thresh = [speed_thresh], # use this to categorize running and immobility
                                        cmap_name='Spectral_r',
                                        kwargs_scatter = kwargs_scatter,
                                        marker_per_speed_category = ['^','o'],
                                        do_plot_maze=True,
                                        position_tsdf=position_tsdf,ds=5,
                                                seperate_colorbar=False,
                                                color_time=color_time
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

