import numpy as np
import pandas as pd

def get_contrast_axis_and_proj(x_sub,tuning,map_state_pre,map_state_post,map_state_win=3):
    '''
    given tuning, get the difference in the population vectors between two states and normalize, 
    project the PV on this contrastive axis

    each axis is averaged within a window (-map_state_win,+map_state_win) to account for sparse firing using the similarity of adjacent states
    '''
    state_ind_pre_range = slice(map_state_pre-map_state_win,map_state_pre+map_state_win+1)
    axis_pre=tuning[state_ind_pre_range].mean(axis=0)

    state_ind_post_range = slice(map_state_post-map_state_win,map_state_post+map_state_win+1)
    axis_post=tuning[state_ind_post_range].mean(axis=0)

    axis_pre_minus_post=axis_pre-axis_post
    axis_pre_minus_post_norm = axis_pre_minus_post/np.linalg.norm(axis_pre_minus_post)
    contrast_axis=axis_pre_minus_post_norm

    proj_on_contrast_axis=x_sub.dot(contrast_axis)
    
    return proj_on_contrast_axis,contrast_axis

# for each trial, segment the trial into chunks with continuous dynamics, seperated by jumps or periods of jumps
# dominant latent within each continuous segment
# neurons tuned to those latents

def segment_trial_by_jump(jump_p_sub,post_map_sub,jump_p_merge_threshold_time=1,is_jump_threshold=0.5):
    '''
    jump_p_sub: n_time
    post_map_sub: n_time
    jump_p_merge_threshold_ind: merge jump if gap between consecutive jump is less than this threshold
    if jump_p_sub is probability, then threshold by is_jump_threshold to get is_jump; otherwise jump_p_sub is boolean is_jump, is_jump_threshold is not used
    '''
    
    
    # merge jump if gap between consecutive jump is less than jump_p_merge_threshold_ind
    jump_epoch = jump_p_sub.threshold(is_jump_threshold).time_support.merge_close_intervals(jump_p_merge_threshold_time)
    continuous_epoch = post_map_sub.time_support.set_diff(jump_epoch)
    
    post_map_median_per_epoch = {}
    for ii,epoch in enumerate(continuous_epoch):
        post_map_median = np.nanmedian(post_map_sub.restrict(epoch))
        post_map_median_per_epoch[ii] = post_map_median
    
    res = {'post_map_median_per_epoch':post_map_median_per_epoch,'jump_epoch':jump_epoch,'continuous_epoch':continuous_epoch}
    return res