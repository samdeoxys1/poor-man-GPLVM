'''
analyze latent (especially in relation to behavior)
e.g. score and classify latent by behavior
'''
import numpy as np
import pandas as pd
import pynapple as nap


def compute_mean_latent_per_behavior(p_latent_marg_beh, ep_d):
    mean_latent_per_beh = {}
    for k, ep in ep_d.items():
        mean_latent_per_beh[k] = p_latent_marg_beh.restrict(ep).mean(axis=0)
    return pd.DataFrame(mean_latent_per_beh)

def classify_latents_by_behavior(p_latent_marg_beh, ep_d, thresh=0.5):
    """
    thresh: float (same for all categories) or dict category_key -> value e.g. {'locomotion': 0.5, 'offmaze': 0.6}.
    Categories not in dict use 0.5.
    """
    mean_latent_per_beh = compute_mean_latent_per_behavior(p_latent_marg_beh, ep_d)
    latent_score_df = mean_latent_per_beh / mean_latent_per_beh.sum(axis=1).values[:, None]
    if isinstance(thresh, dict):
        thresh_d = thresh
        default_t = 0.5
    else:
        thresh_d = None
        default_t = float(thresh) if np.isscalar(thresh) else 0.5
    latent_type_ma_d = {}
    for col in latent_score_df.columns:
        t = thresh_d.get(col, default_t) if thresh_d is not None else default_t
        latent_type_ma_d[col] = latent_score_df[col] >= t
    any_classified = pd.concat([latent_type_ma_d[col] for col in latent_score_df.columns], axis=1).any(axis=1)
    latent_type_ma_d['unclassified'] = ~any_classified
    latent_type_df = pd.concat(latent_type_ma_d, axis=1)
    return latent_type_df, latent_score_df