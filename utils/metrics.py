import numpy as np
import pandas as pd

def compute_hypervolume_2d(df, obj_acc='acc_valid', obj_params='params_k', 
                           ref_point=(70.0, 1600.0), normalize=True,
                           norm_bounds={'acc_min': 70.0, 'acc_max': 95.0, 'params_min': 50.0, 'params_max': 1600.0}):
    """
    Calcula el Hipervolumen (HV) en 2D para un conjunto de soluciones de Pareto.
    
    Objetivos:
    - obj_acc: Precisión (a maximizar).
    - obj_params: Parámetros en K (a minimizar).
    
    Parámetros:
    - ref_point: (acc_min_ref, params_max_ref) punto de referencia en escala original.
    - normalize: si True, escala ambos objetivos a [0, 1] respecto a norm_bounds.
    """
    if df is None or len(df) == 0:
        return 0.0
    
    # Extraer columnas válidas sin NaNs
    sub = df[[obj_acc, obj_params]].dropna().copy()
    if len(sub) == 0:
        return 0.0
    
    acc_ref, params_ref = ref_point
    # Filtrar solo puntos que dominen al punto de referencia
    valid = sub[(sub[obj_acc] >= acc_ref) & (sub[obj_params] <= params_ref)].copy()
    if len(valid) == 0:
        return 0.0

    if normalize:
        a_min = norm_bounds['acc_min']
        a_max = norm_bounds['acc_max']
        p_min = norm_bounds['params_min']
        p_max = norm_bounds['params_max']
        
        # y: accuracy normalizada (mayor es mejor, 0 a 1)
        valid['y_norm'] = np.clip((valid[obj_acc] - a_min) / (a_max - a_min + 1e-8), 0.0, 1.0)
        # x: compacidad normalizada (menos parámetros = mayor valor, 0 a 1)
        valid['x_norm'] = np.clip((p_max - valid[obj_params]) / (p_max - p_min + 1e-8), 0.0, 1.0)
        
        # Extraer puntos no dominados en espacio normalizado
        pts = valid[['x_norm', 'y_norm']].drop_duplicates().values
    else:
        # Espacio crudo: transformar params a beneficio (params_ref - params)
        valid['x_raw'] = params_ref - valid[obj_params]
        valid['y_raw'] = valid[obj_acc] - acc_ref
        pts = valid[['x_raw', 'y_raw']].drop_duplicates().values

    # Filtrar puntos estrictamente no dominados entre sí
    non_dominated = []
    for i in range(len(pts)):
        dominated = False
        for j in range(len(pts)):
            if i == j: continue
            # j domina a i si es >= en ambos y > en al menos uno
            if (pts[j][0] >= pts[i][0] and pts[j][1] >= pts[i][1] and 
               (pts[j][0] > pts[i][0] or pts[j][1] > pts[i][1])):
                dominated = True
                break
        if not dominated:
            non_dominated.append(pts[i])
            
    if len(non_dominated) == 0:
        return 0.0
        
    non_dominated = np.array(non_dominated)
    # Ordenar por x ascendente
    order = np.argsort(non_dominated[:, 0])
    pts_sorted = non_dominated[order]
    
    # Calcular suma de áreas rectangulares
    hv = pts_sorted[0, 0] * pts_sorted[0, 1]
    for i in range(1, len(pts_sorted)):
        dx = pts_sorted[i, 0] - pts_sorted[i - 1, 0]
        # La altura en este intervalo es y del punto actual
        hv += dx * pts_sorted[i, 1]
        
    return float(hv)
