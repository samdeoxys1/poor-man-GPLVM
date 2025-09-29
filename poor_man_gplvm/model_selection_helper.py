'''
Helper functions for model selection
'''
import itertools
import pandas as pd
from typing import Dict, List, Any
from poor_man_gplvm import PoissonGPLVMJump1D,GaussianGPLVMJump1D,PoissonGPLVM1D,GaussianGPLVM1D
import jax
import jax.random as jr 
import numpy as np
import jax.numpy as jnp
import tqdm

model_class_dict = {'poisson':PoissonGPLVMJump1D,'gaussian':GaussianGPLVMJump1D,'poisson_latentonly':PoissonGPLVM1D,'gaussian_latentonly':GaussianGPLVM1D}

default_fit_kwargs = {'n_iter':20,'log_posterior_init':None,'n_time_per_chunk':10000,'dt':1.,'likelihood_scale':1.,'save_every':None,'posterior_init_kwargs':{'random_scale':0.1}}

def generate_hyperparam_grid(hyperparam_ranges: Dict[str, List]) -> List[Dict[str, Any]]:
    """
    Convert dict of lists to list of all combinations
    
    Args:
        hyperparam_ranges: {'param1': [val1, val2], 'param2': [val3, val4]}
        
    Returns:
        List of dicts: [{'param1': val1, 'param2': val3}, ...]
        And the DataFrame version
    """
    keys = list(hyperparam_ranges.keys())
    value_combinations = itertools.product(*[hyperparam_ranges[k] for k in keys])
    hyper_grid_l = [dict(zip(keys, combo)) for combo in value_combinations]
    hyper_grid_df = pd.DataFrame(hyper_grid_l)
    return hyper_grid_l,hyper_grid_df

def fit_model_one_config(config,y_train,key=jr.PRNGKey(0),fit_kwargs=default_fit_kwargs,model_class_str='poisson',n_repeat = 1):
    '''
    create and fit the model with the given config
    fit_kwargs: dict of kwargs for the fit_em function
    n_repeat: number of times to repeat the fitting

    model_class_str: 'poisson' or 'gaussian' or 'poisson_latentonly' or 'gaussian_latentonly'

    return a list of model fits
    '''
    model_fit_l = []
    if model_class_str not in model_class_dict:
        raise ValueError(f"Invalid model class: {model_class_str}")
    model_class = model_class_dict[model_class_str]
    em_res_l = []
    if isinstance(key,list): # can specify a list of jr keys for more control
        key_l = key
    else:
        key_l = jr.split(key,n_repeat)
    for key in key_l:
        
        model_fit = model_class(n_neuron=y_train.shape[1],**config)
        em_res=model_fit.fit_em(y_train,hyperparam={},key=key,**fit_kwargs) # hyperparam is empty because it is already in the initialization
        em_res_l.append(em_res)
        model_fit_l.append(model_fit)
    return model_fit_l,em_res_l

def evaluate_model_one_config(model_fit_l,y_test,key=jr.PRNGKey(1),n_time_per_chunk=10000,latent_downsample_frac=[0.2,0.4,0.6,0.8],downsample_n_repeat=10,metric_type_l=['log_marginal_test','log_one_step_predictive_marginal_test','downsampled_lml','jump_consensus'],jump_dynamics_index=1,jump_consensus_window_size=5,jump_consensus_jump_p_thresh=0.4,jump_consensus_consensus_thresh=0.8):
    '''
    evaluate the fitted model on the test data

    result include:
    - metric_type_i for each model
    - best metric_type_i for each type of metric, and index of the model that achieves it
    - overall metric for each model
    - best overall metric
    - best model index
    '''
    model_eval_result = {}

    decoding_res_l =[]
    for model_fit in model_fit_l:
        decoding_res = model_fit.decode_latent(y_test,n_time_per_chunk=n_time_per_chunk)
        decoding_res_l.append(decoding_res)

    # metric: log_marginal_test
    if 'log_marginal_test' in metric_type_l:
        model_eval_result['log_marginal_test'] = {'value_per_fit':[],'best_value':None,'best_index':None}
        for decoding_res in decoding_res_l:
            model_eval_result['log_marginal_test']['value_per_fit'].append(decoding_res['log_marginal_final'])
        model_eval_result['log_marginal_test']['value_per_fit'] = np.array(model_eval_result['log_marginal_test']['value_per_fit'])

    if 'log_one_step_predictive_marginal_test' in metric_type_l:
        model_eval_result['log_one_step_predictive_marginal_test'] = {'value_per_fit':[],'best_value':None,'best_index':None}
        for decoding_res in decoding_res_l:
            model_eval_result['log_one_step_predictive_marginal_test']['value_per_fit'].append(decoding_res['log_one_step_predictive_marginals_all'].sum())
        model_eval_result['log_one_step_predictive_marginal_test']['value_per_fit'] = np.array(model_eval_result['log_one_step_predictive_marginal_test']['value_per_fit'])
    
    # metric: downsampled_lml
    if 'downsampled_lml' in metric_type_l:
        for downsample_frac in latent_downsample_frac:
            model_eval_result['downsampled_lml_'+str(downsample_frac)] = {'value_per_fit':[],'best_value':None,'best_index':None}
            for model_fit in model_fit_l:
                ds_lml_result = get_downsampled_lml(model_fit,y_test,downsample_frac=downsample_frac,n_repeat=downsample_n_repeat,key=key)
                model_eval_result['downsampled_lml_'+str(downsample_frac)]['value_per_fit'].append(ds_lml_result['value'])
            model_eval_result['downsampled_lml_'+str(downsample_frac)]['value_per_fit'] = np.array(model_eval_result['downsampled_lml_'+str(downsample_frac)]['value_per_fit'])
        

    if 'jump_consensus' in metric_type_l:
        if isinstance(jump_consensus_window_size,int):
            model_eval_result['jump_consensus'] = {'value_per_fit':[],'best_value':None,'best_index':None}
            jump_p_all_chain = []
            for decoding_res in decoding_res_l:
                jump_p = decoding_res['posterior_dynamics_marg'][:,jump_dynamics_index]
                jump_p_all_chain.append(jump_p)
            jump_p_all_chain = np.array(jump_p_all_chain).T # n_time x n_chain
            for jump_p in jump_p_all_chain.T:
                frac_consensus,_,_ = get_jump_consensus(jump_p,jump_p_all_chain,window_size=jump_consensus_window_size,jump_p_thresh = jump_consensus_jump_p_thresh,consensus_thresh=jump_consensus_consensus_thresh)
                model_eval_result['jump_consensus']['value_per_fit'].append(frac_consensus)
            model_eval_result['jump_consensus']['value_per_fit'] = np.array(model_eval_result['jump_consensus']['value_per_fit'])
        elif isinstance(jump_consensus_window_size,list):
            for window_size in jump_consensus_window_size:
                jump_p_all_chain = []
                model_eval_result['jump_consensus_'+str(window_size)] = {'value_per_fit':[],'best_value':None,'best_index':None}
                for decoding_res in decoding_res_l:
                    jump_p = decoding_res['posterior_dynamics_marg'][:,jump_dynamics_index]
                    jump_p_all_chain.append(jump_p)
                jump_p_all_chain = np.array(jump_p_all_chain).T # n_time x n_chain
                for jump_p in jump_p_all_chain.T:
                    frac_consensus,_,_ = get_jump_consensus(jump_p,jump_p_all_chain,window_size=window_size,jump_p_thresh = jump_consensus_jump_p_thresh,consensus_thresh=jump_consensus_consensus_thresh)
                    model_eval_result['jump_consensus_'+str(window_size)]['value_per_fit'].append(frac_consensus)
                model_eval_result['jump_consensus_'+str(window_size)]['value_per_fit'] = np.array(model_eval_result['jump_consensus_'+str(window_size)]['value_per_fit'])
        else:
            print(f"jump_consensus_window_size {jump_consensus_window_size} is not supported")

    # metric_overall
    model_eval_result['metric_overall'] = {'value_per_fit':[],'best_value':None,'best_index':None}
    # model_eval_result['metric_overall']['value_per_fit'] = model_eval_result['log_marginal_test']['value_per_fit']
    # model_eval_result['metric_overall']['value_per_fit'] = model_eval_result['downsampled_lml']['value_per_fit']
    value_per_fit = np.zeros(len(model_fit_l))
    for downsample_frac in latent_downsample_frac:
        value_per_fit += model_eval_result['downsampled_lml_'+str(downsample_frac)]['value_per_fit']
    value_per_fit /= len(latent_downsample_frac)
    model_eval_result['metric_overall']['value_per_fit'] = value_per_fit

    for k in model_eval_result.keys():
        model_eval_result[k]['best_value'] = np.max(model_eval_result[k]['value_per_fit'])
        model_eval_result[k]['best_index'] = np.argmax(model_eval_result[k]['value_per_fit'])
    return model_eval_result

def model_selection_one_split(y,hyperparam_dict,train_index=None,test_index=None,test_frac=0.2,key = jr.PRNGKey(0),model_to_return_type='best_overall',fit_kwargs=default_fit_kwargs,model_class_str='poisson',n_repeat = 5,latent_downsample_frac=[0.2,0.4,0.6,0.8],downsample_n_repeat=10,metric_type_l=['log_marginal_test','log_one_step_predictive_marginal_test','downsampled_lml','jump_consensus'],jump_dynamics_index=1,jump_consensus_window_size=5,jump_consensus_jump_p_thresh=0.4,jump_consensus_consensus_thresh=0.8):
    '''
    for one split of data, fit and evaluate the models given by all configs
    hyperparam_dict: dict of hyperparam ranges
    train_index: index of the train data
    test_index: index of the test data
    test_frac: fraction of the data to be used as test data
    key: random key
    model_to_return_type: 
        - 'best_overall' : return the best model overall
        - 'best_per_config' : return the best model for each config
        - 'all' : return all models
        - 'best_config' : return all models for the best config
    
    fit_kwargs: dict of kwargs for the fit_em function, see core.AbstractGPLVM.fit_em
    model_class_str: 'poisson' or 'gaussian' or 'poisson_latentonly' or 'gaussian_latentonly'
    n_repeat: number of times to repeat the fitting
    latent_downsample_frac: list of downsample fractions for the latent space
    downsample_n_repeat: number of times to repeat the downsampling
    
    return:
    model_selection_res = Dict
    - model_to_return_l: depending on model_to_return_type
    - best_config: the best config
    - best_model: the best model
    - best_model_l: best config, all models
    - model_eval_result_all_configs: the evaluation result for all configs
    '''
    
    T,n_neuron = y.shape
    
    # by default split the data in two contiguous chunks; TODO: make decoder more flexible to take other splits
    if train_index is None:
        train_index = slice(0,int(T*(1-test_frac)))
    if test_index is None:
        test_index = slice(int(T*(1-test_frac)),T)
    y_train = jnp.array(y[train_index])
    y_test = jnp.array(y[test_index])
    hyperparam_grid_l,hyperparam_grid_df = generate_hyperparam_grid(hyperparam_dict)
    model_eval_result_all_configs = {}

    best_model = None
    best_model_l = None
    model_to_return_l = []
    metric_overall_best = -np.inf

    if 'log_posterior_init' in fit_kwargs:
        if fit_kwargs['log_posterior_init'] is not None:
            fit_kwargs['log_posterior_init'] =fit_kwargs['log_posterior_init'][train_index]
    
    for ii,param_dict in enumerate(hyperparam_grid_l): 
        print('== Config {} of {} =='.format(ii+1,len(hyperparam_grid_l)))
        key,_ = jr.split(key)
        key_fit,key_eval = jr.split(key)
        
        model_fit_l,em_res_l = fit_model_one_config(param_dict,y_train,key=key_fit,fit_kwargs=fit_kwargs,model_class_str=model_class_str,n_repeat=n_repeat)
        model_eval_result = evaluate_model_one_config(model_fit_l,y_test,key=key_eval,latent_downsample_frac=latent_downsample_frac,downsample_n_repeat=downsample_n_repeat,metric_type_l=metric_type_l,jump_dynamics_index=jump_dynamics_index,jump_consensus_window_size=jump_consensus_window_size,jump_consensus_jump_p_thresh=jump_consensus_jump_p_thresh,jump_consensus_consensus_thresh=jump_consensus_consensus_thresh)
        # append the best metrics to the result
        if model_eval_result_all_configs == {}:
            for k in model_eval_result.keys():
                model_eval_result_all_configs[k+'_best_value'] = []
                model_eval_result_all_configs[k+'_best_index'] = []
        for k in model_eval_result.keys():
            model_eval_result_all_configs[k+'_best_value'].append(model_eval_result[k]['best_value'])
            model_eval_result_all_configs[k+'_best_index'].append(model_eval_result[k]['best_index'])
        metric_overall_best_value_current = model_eval_result['metric_overall']['best_value']
        
        # if metric_overall is the best, update models and config
        if metric_overall_best_value_current > metric_overall_best:
            metric_overall_best = metric_overall_best_value_current
            best_model = model_fit_l[model_eval_result['metric_overall']['best_index']]
            best_model_l = model_fit_l
            best_config = param_dict
        
        # append models for return
        if model_to_return_type == 'best_per_config':
            model_to_return_l.append(model_fit_l[model_eval_result['metric_overall']['best_index']])
        elif model_to_return_type == 'all':
            model_to_return_l.append(model_fit_l)
    
    if model_to_return_type == 'best_overall':
        model_to_return_l = [best_model]
    elif model_to_return_type == 'best_config':
        model_to_return_l = [best_model_l]
    model_eval_result_all_configs = pd.DataFrame(model_eval_result_all_configs).join(hyperparam_grid_df)
    hyperparam_tosweep_keys = hyperparam_grid_df.columns
    
    model_selection_res = {'model_to_return_l':model_to_return_l,'best_config':best_config,'best_model':best_model,'best_model_l':best_model_l,'model_eval_result_all_configs':model_eval_result_all_configs,'hyperparam_grid_df':hyperparam_grid_df,'hyperparam_tosweep_keys':hyperparam_tosweep_keys}
    

    return model_selection_res

# additional metrics
# metric result is a dict with value and additional keys
def get_downsampled_lml(model_fit,y_test,downsample_frac=0.2,n_repeat=10,key=jr.PRNGKey(4),**kwargs):
    '''
    downsampled log marginal likelihood; downsample the latent space to penalize model complexity
    kwargs see core.AbstractGPLVM.decode_latent
    '''
    key_l = jr.split(key,n_repeat)
    lml_l = []
    n_latent_to_select = int(model_fit.n_latent_bin * downsample_frac)
    for key in key_l:
        # generate latent mask with n_latent_bin length and only n_latent_to_select are 1
        latent_mask = jnp.zeros(model_fit.n_latent_bin)
        latent_mask = latent_mask.at[jr.choice(key,model_fit.n_latent_bin,shape=(n_latent_to_select,),replace=False)].set(1)
        decoding_res = model_fit.decode_latent(y_test,ma_latent=latent_mask,**kwargs)
        lml_l.append(decoding_res['log_marginal_final'])
    ds_lml_mean = np.mean(lml_l)
    ds_lml_std = np.std(lml_l)
    ds_lml_result = {'value':ds_lml_mean,'std':ds_lml_std}
    return ds_lml_result


# jump consensus -- measure the consistency of jumps of all chains relative to one chain
def get_jump_consensus(jump_p,jump_p_all_chain,window_size=5,jump_p_thresh = 0.4,consensus_thresh=0.8):
    '''
    jump_p: jump probability of the best fit, n_time
    jump_p_all_chain: jump probability of all fits, n_time x n_chain
    window_size: window size to check for consistency; is sensitive to the time unit of the bin
    jump_p_thresh: threshold for a jump

    start with the best fit, find the jumps, 
    then check if the jumps are consistent across chains, i.e. a jump occurs within a window in most other chains ( > consensus_thresh)

    frac_consensus: fraction of jumps that are consistent across chains
    is_jump_filtered: whether the jump is consistent across chains, n_time
    whether_consensus_ma: whether the jump is consistent across chains, n_time
    '''

    jump_time_index = np.nonzero(jump_p >= jump_p_thresh)[0]

    # only keep the jump that is common to all chains; ie for each jump there's some jump within some window for other chains
    jump_time_index_consensus = []
    whether_consensus_ma = []
    for jti in jump_time_index:
        # whether_consensus=(jump_p_all_chain[jti-window_size:jti+window_size,:] > jump_p_thresh).any(axis=0).all()
        whether_consensus=(jump_p_all_chain[jti-window_size:jti+window_size,:] > jump_p_thresh).any(axis=0).mean()>=consensus_thresh # instead of requiring all chains, require a certain fraction of chains to have a jump
        whether_consensus_ma.append(whether_consensus)
        if whether_consensus:
            jump_time_index_consensus.append(jti)
    jump_time_index_consensus= np.array(jump_time_index_consensus,dtype=int)
    whether_consensus_ma = np.array(whether_consensus_ma)

    frac_consensus = whether_consensus_ma.mean()

    is_jump_filtered = np.zeros(len(jump_p))
    if len(jump_time_index_consensus) > 0:
        is_jump_filtered[jump_time_index_consensus] = 1

    return frac_consensus,is_jump_filtered,whether_consensus_ma


def get_jump_consensus_shuffle(jump_p, jump_p_all_chain, chain_index, n_shuffle=1000, window_size=5, jump_p_thresh=0.4, consensus_thresh=0.8, key=jr.PRNGKey(42)):
    '''
    Shuffle test version of get_jump_consensus.
    
    jump_p: jump probability of the best fit, n_time
    jump_p_all_chain: jump probability of all fits, n_time x n_chain
    chain_index: index of the chain corresponding to jump_p within jump_p_all_chain
    n_shuffle: number of shuffle iterations
    window_size: window size to check for consistency
    jump_p_thresh: threshold for a jump
    consensus_thresh: threshold for consensus (will be adjusted for excluding the reference chain)
    key: random key for shuffling
    
    Returns:
        dict with keys:
        - 'frac_consensus_distribution': array of frac_consensus values from shuffles
        - 'percentile_2_5': 2.5th percentile of the distribution
        - 'percentile_97_5': 97.5th percentile of the distribution
        - 'mean': mean of the distribution
        - 'std': standard deviation of the distribution
    '''
    
    # Convert inputs to JAX arrays for vectorization
    jump_p = jnp.array(jump_p)
    jump_p_all_chain = jnp.array(jump_p_all_chain)
    
    # Remove the reference chain from jump_p_all_chain for shuffling
    other_chains_mask = jnp.arange(jump_p_all_chain.shape[1]) != chain_index
    jump_p_other_chains = jump_p_all_chain[:, other_chains_mask]
    
    n_time, n_other_chains = jump_p_other_chains.shape
    
    # Generate all random shift amounts at once for vectorization
    # Split keys for each shuffle
    shuffle_keys = jr.split(key, n_shuffle)
    
    # For each shuffle, generate shift amounts for all chains
    def generate_shifts_for_shuffle(shuffle_key):
        chain_keys = jr.split(shuffle_key, n_other_chains)
        # Use vmap to apply randint to each chain key
        return jax.vmap(lambda k: jr.randint(k, shape=(), minval=0, maxval=n_time))(chain_keys)
    
    # Vectorize over all shuffles: shape (n_shuffle, n_other_chains)
    shift_amounts = jax.vmap(generate_shifts_for_shuffle)(shuffle_keys)
    
    # Vectorized circular shift using advanced indexing
    # Create indices for all shifts at once
    time_indices = jnp.arange(n_time)  # shape: (n_time,)
    # Broadcast to create shifted indices: shape (n_shuffle, n_other_chains, n_time)
    shifted_indices = (time_indices[None, None, :] - shift_amounts[:, :, None]) % n_time
    
    # Apply shifts to all chains and shuffles at once: shape (n_shuffle, n_other_chains, n_time)
    shuffled_other_chains = jump_p_other_chains[shifted_indices, jnp.arange(n_other_chains)[None, :, None]]
    
    # Transpose to get the right shape: (n_shuffle, n_time, n_other_chains)
    shuffled_other_chains = shuffled_other_chains.transpose(0, 2, 1)
    
    # Reconstruct full shuffled arrays for all shuffles at once
    # Shape: (n_shuffle, n_time, n_total_chains)
    n_total_chains = jump_p_all_chain.shape[1]
    shuffled_all_chains = jnp.zeros((n_shuffle, n_time, n_total_chains))
    
    # Set reference chain (same for all shuffles)
    shuffled_all_chains = shuffled_all_chains.at[:, :, chain_index].set(jump_p[None, :])
    # Set shuffled other chains
    shuffled_all_chains = shuffled_all_chains.at[:, :, other_chains_mask].set(shuffled_other_chains)
    
    # Efficient consensus calculation - vectorized but avoiding dynamic slicing issues
    # Find jump time points for the reference chain
    is_jump = jump_p >= jump_p_thresh  # shape: (n_time,)
    jump_time_indices = jnp.where(is_jump)[0]
    n_jumps = len(jump_time_indices)
    
    if n_jumps == 0:
        # No jumps found, return zero consensus for all shuffles
        frac_consensus_distribution = jnp.zeros(n_shuffle)
    else:
        # Process each jump separately but vectorize across shuffles
        consensus_results_per_jump = []
        
        for jump_idx in jump_time_indices:
            # Define window bounds (now these are static)
            start_idx = max(0, int(jump_idx) - window_size)
            end_idx = min(n_time, int(jump_idx) + window_size + 1)
            
            # Extract window data for all shuffles: shape (n_shuffle, window_length, n_chains)
            window_data = shuffled_all_chains[:, start_idx:end_idx, :]
            
            # Check if each chain has any jump in the window for each shuffle
            # Shape: (n_shuffle, n_chains)
            chain_has_jump = jnp.any(window_data > jump_p_thresh, axis=1)
            
            # Calculate consensus fraction for each shuffle
            # Shape: (n_shuffle,)
            consensus_fractions = jnp.mean(chain_has_jump, axis=1)
            
            # Check if consensus threshold is met for each shuffle
            # Shape: (n_shuffle,)
            has_consensus = consensus_fractions >= consensus_thresh
            consensus_results_per_jump.append(has_consensus)
        
        # Stack results and calculate mean consensus across jumps
        # Shape: (n_jumps, n_shuffle) -> (n_shuffle,)
        all_consensus_results = jnp.stack(consensus_results_per_jump, axis=0)
        frac_consensus_distribution = jnp.mean(all_consensus_results, axis=0)
    
    # Calculate statistics
    percentile_2_5 = jnp.percentile(frac_consensus_distribution, 2.5)
    percentile_97_5 = jnp.percentile(frac_consensus_distribution, 97.5)
    mean_val = jnp.mean(frac_consensus_distribution)
    std_val = jnp.std(frac_consensus_distribution)
    
    return {
        'frac_consensus_distribution': np.array(frac_consensus_distribution),
        'percentile_2_5': float(percentile_2_5),
        'percentile_97_5': float(percentile_97_5),
        'mean': float(mean_val),
        'std': float(std_val)
    }



def get_lml_test_history(y_test,model,tuning_saved,do_nb=True,ma_temporal=None):
    '''
    with ma_temporal, can specify the temporal mask to be used for the test data; by expanding a full ma_neuron
    '''
    y_test= y_test
    if ma_temporal is not None:
        ma_neuron_ = jnp.ones(y_test.shape[1])
        ma_neuron = ma_neuron_[None,:] * ma_temporal[:,None]
    else:
        ma_neuron = None

    lml_test_l=[]
    for tun_ in tuning_saved:
        if do_nb:
            nb_test_res=model.decode_latent_naive_bayes(y_test,tuning=tun_,ma_neuron=ma_neuron)
            lml_test_l.append(nb_test_res['log_marginal_total'])
        else:
            test_res=model.decode_latent(y_test,tuning=tun_,ma_neuron=ma_neuron)
            lml_test_l.append(test_res['log_marginal_final'])
            
    lml_test_l=np.array(lml_test_l)
    return lml_test_l
    