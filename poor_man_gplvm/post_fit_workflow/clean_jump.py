'''
After fitting the model
Refit transition probability based on (unsmoothed) tuning similarity
Then re-decode
'''
import poor_man_gplvm.transition_analysis as pta
import poor_man_gplvm.fit_tuning_helper as fth
import poor_man_gplvm.gp_kernel as gpk
import numpy as np

def compute_clean_transition_and_decode(
    spk_mat,
    model_fit,
    decode_res=None,
    inverse_temperature_l=np.arange(0, 30),
    metric='cosine',
):
    '''
    Compute tuning-similarity-based transition matrix and decode with it.
    
    Parameters
    ----------
    spk_mat : nap.TsdFrame
        Spike count matrix (n_time x n_neuron)
    model_fit : PoissonGPLVMJump1D
        Fitted model
    decode_res : dict
        Decoding result from model_fit.decode_latent
    posterior_latent_marg : nap.TsdFrame
        Posterior latent marginal (n_time x n_latent_bin)
    inverse_temperature_l : array-like
        Candidate inverse temperatures to search
    metric : str
        Distance metric for tuning similarity
        
    Returns
    -------
    dict with:
        transition_matrix : (n_latent x n_latent) selected transition matrix
        best_inverse_temperature : float
        loss_l : pd.Series, loss vs inverse_temperature
        decode_res_clean : dict, decoding result with clean transition
        posterior_dynamics_marg_clean : nap.TsdFrame
        tc_post : empirical tuning from posterior
        p_trans_target : target transition probability (from continuous dynamics)
        latent_transition_kernel_prior : prior transition kernel
    '''
    if decode_res is None:
        decode_res = model_fit.decode_latent(spk_mat)
    posterior_latent_marg = decode_res['posterior_latent_marg']
    # Empirical tuning from posterior
    tc_post = fth.empirical_tuning_from_posterior(posterior_latent_marg.d, spk_mat.d)
    
    # Target transition: posterior under continuous dynamics only
    p_trans_target = decode_res['p_transition_full'][0, 0] / decode_res['p_transition_full'][0, 0].sum(axis=1, keepdims=True)
    
    # Select inverse temperature
    transition_matrix, best_inverse_temperature, loss_l = pta.select_inverse_temperature(
        tc_post, p_trans_target, inverse_temperature_l=inverse_temperature_l, metric=metric
    )
    
    # Decode with clean transition
    decode_res_clean = model_fit.decode_latent(spk_mat, custom_transition_kernel=transition_matrix)
    posterior_dynamics_marg_clean = decode_res_clean['posterior_dynamics_marg']
    
    # Prior transition kernel
    latent_transition_kernel_prior, _ = gpk.create_transition_prob_latent_1d(
        model_fit.possible_latent_bin, model_fit.movement_variance, custom_kernel=None
    )
    
    return {
        'transition_matrix': transition_matrix,
        'best_inverse_temperature': best_inverse_temperature,
        'loss_l': loss_l,
        'decode_res_clean': decode_res_clean,
        'posterior_dynamics_marg_clean': posterior_dynamics_marg_clean,
        'tc_post': tc_post,
        'p_trans_target': p_trans_target,
        'latent_transition_kernel_prior': latent_transition_kernel_prior,
    }