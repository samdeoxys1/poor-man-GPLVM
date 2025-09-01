'''
helper functions for reactivation analysis
'''

import numpy as np
import pynapple as nap
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import jax.numpy as jnp
import tqdm

'''
two types of shuffling: 
    - test whether it's caused by difference in correlation structure or single neuron activation, preserving spike autocorrelation:
        circularly shuffle spikes within each neuron independently; do it seperately for each pre/post epoch
    - test whether the pre-post difference is just random noise (in terms of having this boundary), so just relabel posterior as from pre/post, no need to redecode 
'''

def decode_pre_post(model,spk_mat_d,pre_post_epoch_d=None,decoder_type='naive_bayes',common_ep=None):
    '''
    decode the latent using the model, for each pre/post epoch
    spk_mat_d: dict, key is pre/post, value is spk_mat; or nap.TsdFrame, in which case we need pre_post_epoch_d to restrict the spk_mat
    pre_post_epoch_d: dict, key is pre/post, value is nap.IntervalSet
    common_ep: e.g. NREM, further restrict both pre/post to this epoch
    '''

    if (pre_post_epoch_d is not None) and isinstance(spk_mat_d,nap.TsdFrame):
        assert ('pre' in pre_post_epoch_d) and ('post' in pre_post_epoch_d)
        spk_mat_d = {pre_post:spk_mat_d.restrict(ep) for pre_post,ep in pre_post_epoch_d.items()}
        if common_ep is not None:
            spk_mat_d = {pre_post:spk_mat_d[pre_post].restrict(common_ep) for pre_post in spk_mat_d.keys()}
    else:
        assert ('pre' in spk_mat_d) and ('post' in spk_mat_d)

    post_latent_d = {}
    post_latent_mean_d = {}

    for pre_post,spk_mat_sub in spk_mat_d.items():
        if decoder_type == 'naive_bayes':
            decode_res = model.decode_latent_naive_bayes(jnp.array(spk_mat_sub))
            post_latent_marg = decode_res['posterior_latent']
        elif decoder_type == 'dynamics':
            decode_res = model.decode_latent(jnp.array(spk_mat_sub))
            post_latent_marg = decode_res['posterior_latent_marg']
        else:
            raise ValueError(f"decoder_type {decoder_type} not supported")
        post_latent_d[pre_post] = post_latent_marg
        post_latent_mean_d[pre_post] = post_latent_marg.mean(axis=0)
    post_latent_mean_d['diff'] = post_latent_mean_d['post'] - post_latent_mean_d['pre']
    post_latent_mean_d = pd.DataFrame(post_latent_mean_d,columns=['pre','post','diff'])
    decode_res_pre_post = {'post_latent_d':post_latent_d,'post_latent_mean_d':post_latent_mean_d}

    return decode_res_pre_post
    
def circular_shuffle_spikes_within_epoch_and_decode(model,spk_mat,pre_post_epoch_d,decoder_type='naive_bayes',common_ep=None,n_shuffle=100):
    '''
    within pre/post epoch, circularly shuffle spikes within each neuron independently
    decode the latent
    return mean pre, post, post-pre posterior
    '''
    spk_mat_d = {}
    for pre_post,ep in pre_post_epoch_d.items():
        spk_mat_sub = spk_mat.restrict(ep)
        if common_ep is not None:
            spk_mat_sub = spk_mat_sub.restrict(common_ep)
        spk_mat_d[pre_post] = spk_mat_sub.d
    
    
    post_latent_mean_d_shuffled = {}
    for i in tqdm.trange(n_shuffle):
        spk_mat_shuffled_d = {}
        for pre_post,spk_mat_sub in spk_mat_d.items():
            spk_mat_sub_shuffled = circular_shuffle_column_independently(spk_mat_sub,min_shift=5)
            spk_mat_shuffled_d[pre_post] = spk_mat_sub_shuffled
        decode_res_pre_post = decode_pre_post(model,spk_mat_shuffled_d,decoder_type=decoder_type)
        post_latent_mean_d_shuffled[i]=decode_res_pre_post['post_latent_mean_d']
    post_latent_mean_d_shuffled = pd.concat(post_latent_mean_d_shuffled,axis=0)

        
    return post_latent_mean_d_shuffled

def circular_shuffle_column_independently(spk_mat,min_shift=5):
    '''
    circularly shuffle the columns of the spk_mat independently
    '''
    n_time,n_neuron = spk_mat.shape
    spk_mat_shuffled = spk_mat.copy()
    for j in range(n_neuron):
        spk_mat_shuffled[:,j] = np.roll(spk_mat[:,j],np.random.randint(min_shift,n_time-min_shift))
    return spk_mat_shuffled

def random_reassign_prepost_latent_difference(posterior_latent_d,n_shuffle=100):
    '''
    random reassign the pre/post; compute mean and difference
    '''
    post_concat = np.concatenate(posterior_latent_d.values,axis=0)
    pre_post_l = np.array(list(posterior_latent_d.keys()))
    labal_l =[]
    for k,val in posterior_latent_d.items():
        labal_l.append(np.array([k]*len(val)))
    labal_l = np.concatenate(labal_l)
    post_latent_mean_d_shuffled_l = {}
    for i in tqdm.trange(n_shuffle):
        label_l_shuffled = np.random.permutation(labal_l)   
        post_concat_shuffled = post_concat[label_l_shuffled]
        post_concat_shuffled_d = {k:post_concat_shuffled[label_l_shuffled==k] for k in pre_post_l}
        post_latent_mean_d_shuffled = {}
        for k,val in post_concat_shuffled_d.items():
            mean = val.mean(axis=0)
            post_latent_mean_d_shuffled[k] = mean
        post_latent_mean_d_shuffled['diff'] = post_latent_mean_d_shuffled['post'] - post_latent_mean_d_shuffled['pre']
        post_latent_mean_d_shuffled_l[i] = post_latent_mean_d_shuffled
    post_latent_mean_d_shuffled_l = pd.concat(post_latent_mean_d_shuffled_l,axis=0)
    return post_latent_mean_d_shuffled_l



    