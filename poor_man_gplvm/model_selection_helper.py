'''
Helper functions for model selection
'''
import itertools
import pandas as pd
from typing import Dict, List, Any
from poor_man_gplvm import PoissonGPLVMJump1D,GaussianGPLVMJump1D
import jax.random as jr 
import numpy as np

model_class_dict = {'poisson':PoissonGPLVMJump1D,'gaussian':GaussianGPLVMJump1D}

default_fit_kwargs = {'n_iter':20,'n_time_per_chunk':10000,'dt':1.,'likelihood_scale':1.,'save_every':None,'posterior_init_kwargs':{'random_scale':0.1}}

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

def fit_model_one_config(config,y_train,key=jr.PRNGKey(0),fit_kwargs=default_fit_kwargs,model_class_str='poisson',n_repeat = 1,**kwargs):
    '''
    create and fit the model with the given config
    fit_kwargs: dict of kwargs for the fit_em function
    n_repeat: number of times to repeat the fitting

    return a list of model fits
    '''
    model_fit_l = []
    if model_class_str not in model_class_dict:
        raise ValueError(f"Invalid model class: {model_class_str}")
    model_class = model_class_dict[model_class_str]
    key_l = jr.split(key,n_repeat)
    for key in key_l:
        model_fit = model_class(n_neuron=y_train.shape[1],**config)
        model_fit.fit_em(y_train,hyperparam={},key=key,**fit_kwargs) # hyperparam is empty because it is already in the initialization
        model_fit_l.append(model_fit)
    return model_fit_l

def evaluate_model_one_config(model_fit_l,y_test,key=jr.PRNGKey(1)):
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

    model_eval_result['log_marginal_test'] = {'value_per_fit':[],'best_value':None,'best_index':None}
    for model_fit in model_fit_l:
        log_posterior_all,log_marginal_final,log_causal_posterior_all = model_fit.decode_latent(y_test)
        model_eval_result['log_marginal_test']['value_per_fit'].append(log_marginal_final)
    model_eval_result['log_marginal_test']['value_per_fit'] = np.array(model_eval_result['log_marginal_test']['value_per_fit'])
    
    # for now testing; metric_overall is the same as log_marginal_test
    model_eval_result['metric_overall'] = {'value_per_fit':[],'best_value':None,'best_index':None}
    model_eval_result['metric_overall']['value_per_fit'] = model_eval_result['log_marginal_test']['value_per_fit']

    for k in model_eval_result.keys():
        model_eval_result[k]['best_value'] = np.max(model_eval_result[k]['value_per_fit'])
        model_eval_result[k]['best_index'] = np.argmax(model_eval_result[k]['value_per_fit'])
    return model_eval_result

def model_selection_one_split(y,hyperparam_dict,train_index=None,test_index=None,test_frac=0.2,key = jr.PRNGKey(0),model_to_return_type='best_overall',**kwargs):
    '''
    for one split of data, fit and evaluate the models given by all configs
    model_to_return_type: 
        - 'best_overall' : return the best model overall
        - 'best_per_config' : return the best model for each config
        - 'all' : return all models
        - 'best_config' : return all models for the best config
    
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
    param_grid_l,param_grid_df = generate_hyperparam_grid(hyperparam_dict)
    model_eval_result_all_configs = {}

    best_model = None
    best_model_l = None
    model_to_return_l = []
    metric_overall_best = -np.inf
    for param_dict in param_grid_l: 
        key,_ = jr.split(key)
        key_fit,key_eval = jr.split(key)
        model_fit_l = fit_model_one_config(param_dict,y_train,key=key_fit,**kwargs)
        model_eval_result = evaluate_model_one_config(model_fit_l,y_test,key=key_eval)
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
    
    model_selection_res = {'model_to_return_l':model_to_return_l,'best_config':best_config,'best_model':best_model,'best_model_l':best_model_l,'model_eval_result_all_configs':model_eval_result_all_configs}
    

    return model_selection_res