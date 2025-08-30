'''
For Roman Huszar's T-maze dataset
'''
import numpy as np
import pandas as pd
from sklearn.cluster import dbscan
import pynapple as nap

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

