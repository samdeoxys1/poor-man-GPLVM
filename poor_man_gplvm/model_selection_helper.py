'''
Helper functions for model selection
'''
import itertools
import pandas as pd
from typing import Dict, List, Any
from poor_man_gplvm import PoissonGPLVMJump1D,GaussianGPLVMJump1D,PoissonGPLVM1D,GaussianGPLVM1D
import jax.random as jr 
import numpy as np
import jax.numpy as jnp

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

def model_selection_one_split(y,hyperparam_dict,train_index=None,test_index=None,test_frac=0.2,key = jr.PRNGKey(0),model_to_return_type='best_overall',fit_kwargs=default_fit_kwargs,model_class_str='poisson',n_repeat = 1,latent_downsample_frac=[0.2],downsample_n_repeat=10,metric_type_l=['log_marginal_test','log_one_step_predictive_marginal_test','downsampled_lml','jump_consensus'],jump_dynamics_index=1,jump_consensus_window_size=5,jump_consensus_jump_p_thresh=0.4,jump_consensus_consensus_thresh=0.8):
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
    y_train = y[train_index]
    y_test = y[test_index]
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