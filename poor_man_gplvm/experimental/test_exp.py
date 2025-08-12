from poor_man_gplvm.test import circular_shuffle_data
import tqdm
import numpy as np

def shuffle_and_decode(model,spk_tsdf,n_time_per_chunk=10000,dt_l=1,n_shuffle=100,ep=None,decoder_type='naive_bayes'):
    '''
    shuffle the data and decode the latent
    decoder_type: 'naive_bayes' or 'dynamics'; dynamics is the one used for EM, the bayesian smoother, with different dynamics on latent; 
    '''
    y_shuffled_l = circular_shuffle_data(spk_tsdf,n_shuffle=n_shuffle,ep=ep)
    decoding_res_l = []
    for y_shuffled in tqdm.tqdm(y_shuffled_l,total=n_shuffle):
        model.get_gain_mstep_chunk(y_shuffled,model.log_posterior,model.tuning,n_time_per_chunk=n_time_per_chunk)
        if decoder_type == 'naive_bayes':
            decoding_res = model.decode_latent_naive_bayes(y_shuffled,n_time_per_chunk=n_time_per_chunk,dt_l=dt_l)
        elif decoder_type == 'dynamics':
            decoding_res = model.decode_latent(y_shuffled,n_time_per_chunk=n_time_per_chunk)
        else:
            raise ValueError(f"decoder_type {decoder_type} not supported")
        decoding_res_l.append(decoding_res)
    
    # reshape the decoding_res_l to each key having n_shuffle elements
    decoding_res_l = {k:np.array([d[k] for d in decoding_res_l]) for k in decoding_res_l[0].keys()}
    return decoding_res_l