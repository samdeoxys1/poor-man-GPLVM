'''
Wrap supervised tuning_flat and (log) posterior matrices into xarray with label_bin coords
matching decoder_supervised (MultiIndex from flat_idx_to_coord; optional per-maze split).

get_tuning already returns tuning on the full grid as tuning_res["tuning"]; use
tuning_flat_to_xr for the valid-bin matrix used in decoding.
'''

import numpy as np
import pandas as pd
import xarray as xr


def _label_cols(flat_idx_to_coord):
    return [c for c in flat_idx_to_coord.columns if c != 'maze']


def _multiindex_from_df_lab(df_lab):
    df_lab = df_lab.copy()
    for c in df_lab.columns:
        if df_lab[c].dtype.kind in 'if':
            df_lab[c] = df_lab[c].fillna(-1)
        else:
            df_lab[c] = df_lab[c].fillna('_nan')
    return pd.MultiIndex.from_frame(df_lab)


def _label_bin_coord_for_rows(flat_idx_to_coord, flat_rows):
    '''
    flat_rows: positional indices into tuning_flat rows (same as df.index labels).
    '''
    df = flat_idx_to_coord
    label_cols = _label_cols(df)
    if not label_cols:
        return np.asarray(flat_rows)
    df_sub = df.loc[flat_rows, label_cols]
    return _multiindex_from_df_lab(df_sub)


def tuning_flat_to_xr(tuning_flat, flat_idx_to_coord, neuron_names=None, split_maze=False):
    '''
    Wrap (n_latent, n_neuron) supervised tuning used in decoding.

    If split_maze and multiple mazes in flat_idx_to_coord, returns dict[maze, DataArray];
    otherwise one DataArray with label_bin = MultiIndex over all rows.
    '''
    arr = np.asarray(tuning_flat)
    if arr.ndim != 2:
        raise ValueError(f'tuning_flat must be 2D, got shape {arr.shape}')
    n_latent, n_neuron = arr.shape
    df = flat_idx_to_coord
    if not isinstance(df, pd.DataFrame) or 'maze' not in df.columns:
        raise ValueError("flat_idx_to_coord must be a DataFrame with column 'maze'")
    if len(df) != n_latent:
        raise ValueError(f'len(flat_idx_to_coord)={len(df)} != tuning_flat.shape[0]={n_latent}')

    if neuron_names is None:
        neuron_names = np.arange(n_neuron)

    maze_l = pd.unique(df['maze'])
    if split_maze and len(maze_l) > 1:
        out = {}
        for maze in maze_l:
            idx = np.sort(df.index[df['maze'] == maze].to_numpy())
            lab = _label_bin_coord_for_rows(df, idx)
            out[str(maze)] = xr.DataArray(
                arr[idx, :],
                dims=('label_bin', 'neuron'),
                coords={'label_bin': lab, 'neuron': np.asarray(neuron_names)},
            )
        return out

    lab = _label_bin_coord_for_rows(df, df.index.to_numpy())
    return xr.DataArray(
        arr,
        dims=('label_bin', 'neuron'),
        coords={'label_bin': lab, 'neuron': np.asarray(neuron_names)},
    )


def latent_time_series_to_xr(
    values,
    flat_idx_to_coord,
    time_coord=None,
    event_index_per_bin=None,
    dim_name='value',
    split_maze=False,
):
    '''
    Wrap (n_time, n_latent) arrays (posterior, log posterior, log-likelihood per bin, etc.).

    Column order must match flat_idx_to_coord.index (global flat index), same as decoder output.

    If split_maze and multiple mazes, returns dict[maze, DataArray] with only that maze's
    label_bin columns (same convention as decoder_supervised._wrap_label_results_xr_by_maze).
    '''
    arr = np.asarray(values)
    if arr.ndim != 2:
        raise ValueError(f'values must be 2D, got shape {arr.shape}')
    n_time, n_latent = arr.shape
    df = flat_idx_to_coord
    if not isinstance(df, pd.DataFrame) or 'maze' not in df.columns:
        raise ValueError("flat_idx_to_coord must be a DataFrame with column 'maze'")

    if time_coord is None:
        time_coord = np.arange(n_time)
    else:
        time_coord = np.asarray(time_coord)
    if time_coord.shape[0] != n_time:
        raise ValueError(f'time_coord length {time_coord.shape[0]} != n_time={n_time}')

    coords_time = {'time': time_coord}
    if event_index_per_bin is not None:
        coords_time['event_index_per_bin'] = ('time', np.asarray(event_index_per_bin))

    maze_l = pd.unique(df['maze'])

    def _one_maze(maze):
        flat_idx = df.index[df['maze'] == maze].to_numpy(dtype=int)
        sub = arr[:, flat_idx]
        lab = _label_bin_coord_for_rows(df, flat_idx)
        return xr.DataArray(
            sub,
            dims=('time', 'label_bin'),
            coords=dict(coords_time, label_bin=lab),
            name=dim_name,
        )

    if split_maze and len(maze_l) > 1:
        return {str(m): _one_maze(m) for m in maze_l}

    if len(maze_l) > 1 and not split_maze:
        lab = _label_bin_coord_for_rows(df, df.index.to_numpy())
        if n_latent != len(df):
            raise ValueError(
                f'values has n_latent={n_latent} but flat_idx_to_coord has {len(df)} rows; '
                'use split_maze=True for multi-maze column layout'
            )
        return xr.DataArray(
            arr,
            dims=('time', 'label_bin'),
            coords=dict(coords_time, label_bin=lab),
            name=dim_name,
        )

    lab = _label_bin_coord_for_rows(df, df.index.to_numpy())
    return xr.DataArray(
        arr,
        dims=('time', 'label_bin'),
        coords=dict(coords_time, label_bin=lab),
        name=dim_name,
    )


def supervised_tuning_views(tuning_res):
    '''
    Convenience: grid tuning (already xr) + flat tuning as xarray from tuning_res dict.

    Returns dict with keys: tuning_grid (xr.DataArray), tuning_flat_xr (xr or dict if split).
    '''
    flat_idx_to_coord = tuning_res['flat_idx_to_coord']
    neuron_names = tuning_res.get('neuron_names', None)
    tf = tuning_flat_to_xr(
        tuning_res['tuning_flat'],
        flat_idx_to_coord,
        neuron_names=neuron_names,
        split_maze=False,
    )
    return {
        'tuning_grid': tuning_res['tuning'],
        'tuning_flat_xr': tf,
    }
