'''
Linear maze (e.g. MISI) replay analysis: same as T-maze unsupervised replay
but without offmaze (behavior_ep_d has empty offmaze via offmaze_method='none').
Uses multi-dynamics and skips compare-transition by default.
'''
import os
import poor_man_gplvm.post_fit_workflow.post_fit_tmaze_dataset as pft


def analyze_replay_unsupervised_linear(
    model_fit,
    prep_res,
    pbe_dt=0.02,
    data_dir_full='./',
    force_reload=False,
    dosave=True,
    save_dir=None,
    save_fn='analyze_replay_unsupervised_linear.pkl',
    final_gain=None,
    pbe_kwargs=None,
    behavior_kwargs=None,
    gain_sweep_kwargs=None,
    decode_compare_kwargs=None,
    decode_multiple_transition_kwargs=None,
    nb_decode_kwargs=None,
    use_multi_dynamics=True,
    force_reload_multi_dynamics=False,
    skip_compare_transition=True,
    neuron_indices=None,
    verbose=True,
):
    '''
    Replay analysis for linear maze: no offmaze. Calls analyze_replay_unsupervised
    with offmaze_method='none', use_multi_dynamics=True, skip_compare_transition=True by default.
    prep_res same contract as tmaze (position_tsdf, speed_tsd, full_ep, sleep_state_intervals_NREMepisode,
    ripple_intervals, spk_mat, spk_times).
    '''
    if save_dir is None:
        save_dir = os.path.join(str(data_dir_full), 'py_data', 'analyze_replay_unsupervised_linear')
    behavior_kwargs_ = {'offmaze_method': 'none', 'speed_immo_thresh': 2.5, 'speed_loco_thresh': 5.0}
    if behavior_kwargs is not None:
        behavior_kwargs_.update(dict(behavior_kwargs))
    return pft.analyze_replay_unsupervised(
        model_fit,
        prep_res,
        pbe_dt=pbe_dt,
        data_dir_full=data_dir_full,
        force_reload=force_reload,
        dosave=dosave,
        save_dir=save_dir,
        save_fn=save_fn,
        final_gain=final_gain,
        pbe_kwargs=pbe_kwargs,
        behavior_kwargs=behavior_kwargs_,
        gain_sweep_kwargs=gain_sweep_kwargs,
        decode_compare_kwargs=decode_compare_kwargs,
        decode_multiple_transition_kwargs=decode_multiple_transition_kwargs,
        nb_decode_kwargs=nb_decode_kwargs,
        use_multi_dynamics=use_multi_dynamics,
        force_reload_multi_dynamics=force_reload_multi_dynamics,
        skip_compare_transition=skip_compare_transition,
        neuron_indices=neuron_indices,
        verbose=verbose,
    )
