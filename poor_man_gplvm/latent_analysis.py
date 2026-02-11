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

def classify_latents_by_behavior(p_latent_marg_beh,ep_d, thresh=0.5):
    mean_latent_per_beh = compute_mean_latent_per_behavior(p_latent_marg_beh, ep_d)
    latent_score_df = mean_latent_per_beh / mean_latent_per_beh.sum(axis=1).values[:, None]
    latent_type_ma_d = {}
    for col in latent_score_df.columns:
        latent_type_ma_d[col] = latent_score_df[col] >= thresh
    latent_type_ma_d['unclassified'] = (latent_score_df <= thresh).all(axis=1)
    latent_type_df = pd.concat(latent_type_ma_d, axis=1)
    return latent_type_df,latent_score_df