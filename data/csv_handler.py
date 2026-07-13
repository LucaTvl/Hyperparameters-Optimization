import os
import pandas as pd
from scipy import stats
from config.settings import CSV_GT, CSV_ZCP, CSV_ZCP_FULL, CSV_ZCP_EXT, ZCP_COLS, ZCP_COLS_WITH_PARAMS

def load_ground_truth():
    """
    Carga el ground truth completo con las 15.625 arquitecturas 
    y genera sus rankings reales según acc_valid.
    """
    assert os.path.exists(CSV_GT), f'No encontrado: {CSV_GT}'
    df = pd.read_csv(CSV_GT)
    df['rank_real'] = df['acc_valid'].rank(ascending=False, method='min').astype(int)
    return df

def get_normalization_params(df_full, cols, base_signs=None):
    """
    Calcula los parámetros de normalización (signo y rango).
    Si base_signs es provisto, usa esos signos. Sino, utiliza
    correlación de Spearman contra acc_valid.
    """
    norm_params = {}
    for col in cols:
        if base_signs and col in base_signs:
            sign = base_signs[col]
        else:
            r, _ = stats.spearmanr(df_full[col], df_full['acc_valid'])
            sign = 1 if r >= 0 else -1
        vals = df_full[col] * sign
        vmin, vmax = vals.min(), vals.max()
        norm_params[col] = {'sign': sign, 'min': vmin, 'max': vmax}
    return norm_params

def apply_normalization(df, norm_params, cols):
    """
    Aplica la normalización Min-Max al DataFrame basándose 
    en los parámetros previamente calculados.
    """
    df_norm = df.copy()
    for col in cols:
        sign = norm_params[col]['sign']
        vmin = norm_params[col]['min']
        vmax = norm_params[col]['max']
        vals = df[col] * sign
        df_norm[f'{col}_norm'] = (vals - vmin) / (vmax - vmin + 1e-8)
    return df_norm

def split_calib_val(df_norm):
    """
    Divide el dataset normalizado en conjunto de calibración (1001 arquitecturas) 
    y conjunto de validación, asegurando coincidencia con ejecuciones previas.
    """
    df_calib_ids = pd.read_csv(CSV_ZCP)[['arch_id']]
    df_calib = df_calib_ids.merge(df_norm, on='arch_id')
    df_val = df_norm[~df_norm['arch_id'].isin(df_calib['arch_id'])].copy()
    return df_calib, df_val
