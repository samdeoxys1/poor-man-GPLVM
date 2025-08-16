'''
For Yiyao Zhang's ACH dataset

need:
- transition events
    - continuous and fragmented chunks
        - further subdivide by sleep stage
    - previously tried detecting ACh bouts; however, it can be more or less noisy depending on the session, making detection criteria sensitive to sessions; thus 
    it's easier to detect the model-derived transition
- feature
    - ach, pop fr, consec pv difference, 
- peri transition feature average and shuffle
- tuning
    - jump probability vs sleep stage
- representational analysis 
    - (here the observation is the latent during continuous states tend to be similar within a NREM interval, but can be different across NREM intervals)
    - mean latent posterior 
- 

'''

import numpy as np
import scipy
import pynapple as nap


# helper function to get list of processed decoding result from em_res_l (multiple em fit results)
def get_decode_res_l_from_em_res_l(em_res_l,load_res):
    decode_res_l = []
    for em_res in em_res_l:
        log_posterior_final=em_res['log_posterior_final']
        post_latent_marg = np.exp(scipy.special.logsumexp(log_posterior_final,axis=1))
        post_dynamics_marg = np.exp(scipy.special.logsumexp(log_posterior_final,axis=2))
        
        decode_res_one = {'posterior_latent_marg':nap.TsdFrame(d=post_latent_marg,t=load_res['t_l']),'posterior_dynamics_marg':nap.TsdFrame(d=post_dynamics_marg,t=load_res['t_l'])}
        
        decode_res_l.append(decode_res_one)
    return decode_res_l