import os
import bz2
import pickle
import pandas as pd
import numpy as np
from scipy import stats
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    from tqdm.auto import tqdm
except ImportError:
    from tqdm import tqdm

from config.settings import (
    CSV_GT, CSV_ZCP, CSV_ZCP_FULL, CSV_ZCP_EXT,
    ZCP_COLS, ZCP_COLS_WITH_PARAMS, NATS_PATH,
    DATASET, BUDGET, BENCHMARK_SEED, N_ARCHS, DEVICE
)
from models.zcp_metrics import compute_all_zcps


# ── Extracción y Generación de Ground Truth (NATS-Bench TSS) ──

def load_raw(arch_id: int, nats_path: str = NATS_PATH) -> dict:
    """
    Carga el archivo .pickle.pbz2 correspondiente a una arquitectura de NATS-Bench.
    """
    path = os.path.join(nats_path, f'{arch_id:06d}.pickle.pbz2')
    with bz2.BZ2File(path, 'rb') as f:
        return pickle.load(f)


def get_arch_row(
    arch_id: int,
    nats_path: str = NATS_PATH,
    budget: str = BUDGET,
    dataset: str = DATASET,
    seed: int = BENCHMARK_SEED
) -> dict | None:
    """
    Extrae y formatea las métricas del ground truth de NATS-Bench para una arquitectura dada.
    """
    try:
        raw = load_raw(arch_id, nats_path)
        inner = raw[budget]
        entry = inner['all_results'].get((dataset, seed))
        if entry is None:
            return None

        vk = sorted(
            [k for k in entry['eval_acc1es'] if 'x-valid' in k],
            key=lambda x: int(x.split('@')[1])
        )
        tk = sorted(
            [k for k in entry['eval_acc1es'] if 'ori-test' in k],
            key=lambda x: int(x.split('@')[1])
        )

        params_val = entry['params'].item() if hasattr(entry['params'], 'item') else float(entry['params'])
        flop_val = entry['flop'].item() if hasattr(entry['flop'], 'item') else float(entry['flop'])
        latency_val = entry['latency'][0] if isinstance(entry['latency'], (list, tuple, np.ndarray)) else entry['latency']
        if hasattr(latency_val, 'item'):
            latency_val = latency_val.item()

        # params raw está en millones → convertimos a miles (K)
        return {
            'arch_id'   : arch_id,
            'arch_str'  : inner['arch_str'],
            'params_k'  : round(float(params_val) * 1000, 2),
            'flops_m'   : round(float(flop_val), 4),
            'latency_ms': round(float(latency_val) * 1000, 4),
            'acc_valid' : entry['eval_acc1es'][vk[-1]] if vk else None,
            'acc_test'  : entry['eval_acc1es'][tk[-1]] if tk else None,
        }
    except Exception:
        return None


def generate_ground_truth_csv(
    output_path: str = CSV_GT,
    nats_path: str = NATS_PATH,
    n_archs: int = N_ARCHS,
    max_workers: int = 8
) -> pd.DataFrame:
    """
    Construye el CSV de Ground Truth leyendo en paralelo los archivos crudos
    .pickle.pbz2 de NATS-Bench TSS (lógica portada desde Untitled0).
    """
    if not os.path.exists(nats_path):
        raise FileNotFoundError(
            f"No se encontró el directorio de NATS-Bench en: '{nats_path}'. "
            "Por favor verifica la ruta o define NATS_PATH en config/settings.py."
        )

    print(f'Construyendo {output_path} desde archivos crudos de NATS-Bench (~15 min)...')
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    rows = [None] * n_archs
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(get_arch_row, i, nats_path): i for i in range(n_archs)}
        for future in tqdm(as_completed(futures), total=n_archs, desc="Extrayendo Ground Truth"):
            result = future.result()
            if result:
                rows[result['arch_id']] = result

    df = pd.DataFrame([r for r in rows if r is not None])
    df.to_csv(output_path, index=False)
    print(f'[OK] CSV guardado: {output_path} ({len(df)} arquitecturas)')
    return df


def load_ground_truth(csv_path: str = CSV_GT, nats_path: str = NATS_PATH) -> pd.DataFrame:
    """
    Carga el ground truth completo con las 15.625 arquitecturas y genera sus rankings reales.
    Si el CSV no existe, realiza la carga manual/extracción desde los pickles de NATS-Bench.
    """
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        print(f'[OK] Ground truth cargado desde disco: {csv_path} ({len(df)} arquitecturas)')
    else:
        print(f'[!] CSV Ground Truth no detectado en {csv_path}. Iniciando carga manual...')
        df = generate_ground_truth_csv(output_path=csv_path, nats_path=nats_path)

    df['rank_real'] = df['acc_valid'].rank(ascending=False, method='min').astype(int)
    return df


# ── Extracción y Generación de Zero-Cost Proxies (ZCPs) ──

def generate_zcps_csv(
    df: pd.DataFrame,
    x_batch,
    y_batch,
    output_path: str = CSV_ZCP,
    n_bins: int = 20,
    per_bin: int = 100,
    seed: int = 42
) -> pd.DataFrame:
    """
    Genera el CSV de ZCPs estratificados para calibración (lógica portada desde Untitled0).
    Muestrea según bins de acc_valid y calcula los proxies para cada arquitectura.
    """
    print(f'Construyendo muestra estratificada de ZCPs en {output_path}...')
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    if hasattr(x_batch, 'to'):
        x_batch = x_batch.to(DEVICE)
    if hasattr(y_batch, 'to'):
        y_batch = y_batch.to(DEVICE)
    np.random.seed(seed)
    df_temp = df.copy()
    df_temp['acc_bin'] = pd.cut(df_temp['acc_valid'], bins=n_bins, labels=False)
    sample_ids = (
        df_temp.groupby('acc_bin', observed=True)
        .apply(lambda g: g.sample(min(len(g), per_bin), random_state=seed))
        .reset_index(drop=True)['arch_id'].tolist()
    )
    print(f'Muestra estratificada: {len(sample_ids)} arquitecturas')

    rows_zcp = []
    failed = 0
    for arch_id in tqdm(sample_ids, desc="Computando ZCPs muestra"):
        astr = df.loc[df['arch_id'] == arch_id, 'arch_str'].values[0]
        res = compute_all_zcps(astr, x_batch, y_batch)
        if res:
            res['arch_id'] = arch_id
            rows_zcp.append(res)
        else:
            failed += 1

    df_zcp = pd.DataFrame(rows_zcp)
    df_zcp.to_csv(output_path, index=False)
    print(f'[OK] ZCPs guardados: {len(df_zcp)} arquitecturas ({failed} fallidas) en {output_path}')
    return df_zcp


def load_zcps(
    df: pd.DataFrame = None,
    x_batch = None,
    y_batch = None,
    csv_path: str = CSV_ZCP
) -> pd.DataFrame:
    """
    Carga el CSV de ZCPs estratificados (conjunto de calibración).
    Si el CSV no existe, realiza la carga manual / cómputo estratificado.
    """
    if os.path.exists(csv_path):
        df_zcp = pd.read_csv(csv_path)
        print(f'[OK] ZCPs de calibracion cargados desde disco: {csv_path} ({len(df_zcp)} arquitecturas)')
        return df_zcp
    else:
        print(f'[!] CSV de ZCPs no detectado en {csv_path}. Iniciando carga manual/computo...')
        if df is None:
            df = load_ground_truth()
        if x_batch is None or y_batch is None:
            from data.dataset import get_cifar10_batch
            x_batch, y_batch = get_cifar10_batch(batch_size=64)
        return generate_zcps_csv(df, x_batch, y_batch, output_path=csv_path)


def load_full_zcps(
    df: pd.DataFrame,
    x_batch,
    y_batch,
    csv_path: str = CSV_ZCP_FULL,
    calib_csv: str = CSV_ZCP
) -> pd.DataFrame:
    """
    Carga o genera los ZCPs para las 15.625 arquitecturas completas.
    Si CSV_ZCP_FULL existe, lo carga de disco.
    Si no existe, realiza la carga manual/cómputo reutilizando las arquitecturas
    ya presentes en calib_csv (si existe) y calculando las pendientes.
    """
    if os.path.exists(csv_path):
        df_zcp_full = pd.read_csv(csv_path)
        print(f'[OK] ZCPs completos cargados desde disco: {csv_path} ({len(df_zcp_full)} arquitecturas)')
        return df_zcp_full

    print(f'[!] CSV de ZCPs completos no detectado en {csv_path}. Iniciando carga manual/computo...')
    already_done = set()
    df_prev = pd.DataFrame()
    if os.path.exists(calib_csv):
        df_prev = pd.read_csv(calib_csv)
        already_done = set(df_prev['arch_id'].tolist())

    pending = [i for i in range(len(df)) if i not in already_done]
    print(f'Computando ZCPs con GPU para {len(pending)} arquitecturas pendientes...')
    if hasattr(x_batch, 'to'):
        x_batch = x_batch.to(DEVICE)
    if hasattr(y_batch, 'to'):
        y_batch = y_batch.to(DEVICE)
    new_rows = []
    for arch_id in tqdm(pending, desc="Calculando ZCPs full"):
        astr = df.loc[df['arch_id'] == arch_id, 'arch_str'].values[0]
        res = compute_all_zcps(astr, x_batch, y_batch)
        if res:
            res['arch_id'] = arch_id
            new_rows.append(res)

    df_new = pd.DataFrame(new_rows)
    df_zcp_full = pd.concat([df_prev, df_new], ignore_index=True)
    df_zcp_full = df_zcp_full.sort_values('arch_id').reset_index(drop=True)
    os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
    df_zcp_full.to_csv(csv_path, index=False)
    print(f'[OK] ZCPs completos guardados: {len(df_zcp_full)} arquitecturas en {csv_path}')
    return df_zcp_full


# ── Normalización y Partición de Datos ──

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


def split_calib_val(df_norm, df_calib_ids=None):
    """
    Divide el dataset normalizado en conjunto de calibración (1001 arquitecturas) 
    y conjunto de validación, asegurando coincidencia con ejecuciones previas.
    Si df_calib_ids no es provisto, se asegura de que CSV_ZCP exista (o lo genera)
    antes de filtrar los IDs.
    """
    if df_calib_ids is None:
        if not os.path.exists(CSV_ZCP):
            load_zcps()
        df_calib_ids = pd.read_csv(CSV_ZCP)[['arch_id']]
    elif isinstance(df_calib_ids, pd.DataFrame) and 'arch_id' in df_calib_ids.columns:
        df_calib_ids = df_calib_ids[['arch_id']]

    df_calib = df_calib_ids.merge(df_norm, on='arch_id')
    df_val = df_norm[~df_norm['arch_id'].isin(df_calib['arch_id'])].copy()
    return df_calib, df_val
