import os
import random
import numpy as np
import pandas as pd
import itertools
try:
    from tqdm.auto import tqdm
except ImportError:
    from tqdm import tqdm

from config.settings import CSV_ZCP_EXT, CHANNELS, NUM_CELLS, SAMPLE_PER_GROUP, MASTER_SEED, ZCP_COLS
from models.zcp_metrics import compute_zcps_ext
from models.networks import count_params_k
from optimizers.genetic_ops import random_ind_ext, crossover_ext, mutate_ext, tournament, random_weights, blx_crossover, mutate_gaussian
from optimizers.ga_level1 import weights_to_dict
from optimizers.ga_level2 import genes_to_arch_str
from optimizers.nsga2 import fast_non_dominated_sort, nsga2_select

def decode_ext(genes):
    arch_str = genes_to_arch_str(genes[:6])
    C = CHANNELS[genes[6]]
    nc = NUM_CELLS[genes[7]]
    return arch_str, C, nc

def get_extended_zcps(df_base, x_batch, y_batch):
    if os.path.exists(CSV_ZCP_EXT):
        df_ext = pd.read_csv(CSV_ZCP_EXT)
        print(f'✓ ZCPs extendidos cargados: {len(df_ext)} arquitecturas')
        return df_ext
    
    print('Computando ZCPs para espacio extendido...')
    rows_ext = []
    np.random.seed(MASTER_SEED + 300)
    all_arch_strs = df_base['arch_str'].tolist()

    combinations = list(itertools.product(CHANNELS, NUM_CELLS))
    pbar = tqdm(combinations, desc="Calculando Espacio Extendido")
    
    for C, nc in pbar:
        sample_archs = np.random.choice(all_arch_strs, size=SAMPLE_PER_GROUP, replace=False)
        for astr in sample_archs:
            res = compute_zcps_ext(astr, C, nc, x_batch, y_batch)
            if res:
                res['arch_str'] = astr
                res['channel'] = C
                res['num_cells'] = nc
                res['params_k'] = count_params_k(astr, C, nc)
                rows_ext.append(res)

    df_ext = pd.DataFrame(rows_ext)
    df_ext.to_csv(CSV_ZCP_EXT, index=False)
    print(f'✓ Guardado: {len(df_ext)} arquitecturas en {CSV_ZCP_EXT}')
    return df_ext

def recalibrate_ga1(df_calib_renorm):
    print('\nRe-optimizando GA Nivel 1 con nueva normalización...')
    
    def fitness_weights_renorm(w):
        score = sum(w[i] * df_calib_renorm[f'{col}_norm'] for i, col in enumerate(ZCP_COLS))
        rank_proxy = score.rank(ascending=False, method='min')
        top_real = set(df_calib_renorm.nsmallest(50, 'rank_real')['arch_id'])
        top_proxy = set(df_calib_renorm[rank_proxy <= 50]['arch_id'])
        return len(top_real & top_proxy) / 50

    results_ga1_ext = []
    
    pbar_runs = tqdm(range(10), desc="GA1 Extendido (10 runs)")
    for run in pbar_runs:
        random.seed(MASTER_SEED + run * 17)
        pop = [random_weights(len(ZCP_COLS)) for _ in range(20)]
        fits = [fitness_weights_renorm(ind) for ind in pop]

        for gen in range(5):
            sorted_idx = sorted(range(len(pop)), key=lambda i: fits[i], reverse=True)
            new_pop = [pop[i][:] for i in sorted_idx[:2]]
            while len(new_pop) < 20:
                p1 = tournament(pop, fits)
                p2 = tournament(pop, fits)
                c = blx_crossover(p1, p2) if random.random()<0.8 else p1[:]
                if random.random() < 0.3:
                    c = mutate_gaussian(c, 0.15)
                new_pop.append(c)
            pop = new_pop
            fits = [fitness_weights_renorm(ind) for ind in pop]

        best_idx = max(range(len(pop)), key=lambda i: fits[i])
        results_ga1_ext.append({'best_p50': fits[best_idx], 'weights' : pop[best_idx]})
        pbar_runs.set_postfix(best_p50=f"{fits[best_idx]:.4f}")

    best_run_ext = max(results_ga1_ext, key=lambda r: r['best_p50'])
    return weights_to_dict(best_run_ext['weights'])

def compute_fitness_ext(arch_str, C, num_cells, norm_params_ext, best_weights_ext, x_batch, y_batch):
    res = compute_zcps_ext(arch_str, C, num_cells, x_batch, y_batch)
    if res is None:
        return -np.inf, np.inf

    score = 0.0
    for col in ZCP_COLS:
        val = res[col] * norm_params_ext[col]['sign']
        val_norm = np.clip(
            (val - norm_params_ext[col]['min']) / (norm_params_ext[col]['max'] - norm_params_ext[col]['min'] + 1e-8), 0, 1
        )
        score += best_weights_ext[col] * val_norm

    params = count_params_k(arch_str, C, num_cells)
    p_norm = np.clip(
        (params * norm_params_ext['params_k']['sign'] - norm_params_ext['params_k']['min']) / 
        (norm_params_ext['params_k']['max'] - norm_params_ext['params_k']['min'] + 1e-8), 0, 1
    )
    score += best_weights_ext.get('params_k', 0.0) * p_norm  # Asumiendo params_k podria estar o no
    return score, params

def run_nsga2_extended(norm_params_ext, best_weights_ext, x_batch, y_batch, df_base, pop_size=60, n_gen=30, seed=700):
    print('\nEjecutando NSGA-II en espacio extendido...')
    random.seed(seed)
    
    fitness_cache_ext = {}
    
    def get_objectives_ext(genes):
        key = tuple(genes)
        if key not in fitness_cache_ext:
            astr, C, nc = decode_ext(genes)
            score, params = compute_fitness_ext(astr, C, nc, norm_params_ext, best_weights_ext, x_batch, y_batch)
            fitness_cache_ext[key] = (score, params)
        return fitness_cache_ext[key]

    pop_ext = [random_ind_ext() for _ in range(pop_size)]
    scores_ext = [get_objectives_ext(ind) for ind in pop_ext]
    
    pbar = tqdm(range(n_gen), desc="NSGA-II Extendido")
    for gen in pbar:
        offspring = []
        while len(offspring) < pop_size:
            fronts = fast_non_dominated_sort(pop_ext, scores_ext)
            ranks = [0]*len(pop_ext)
            for r, front in enumerate(fronts):
                for i in front: ranks[i] = r
            
            def sel(k=2):
                cands = random.sample(range(len(pop_ext)), k)
                return pop_ext[min(cands, key=lambda i: ranks[i])][:]
                
            p1, p2 = sel(), sel()
            c1, c2 = crossover_ext(p1,p2) if random.random()<0.8 else (p1[:],p2[:])
            offspring.append(mutate_ext(c1))
            if len(offspring) < pop_size: offspring.append(mutate_ext(c2))

        new_scores = [get_objectives_ext(ind) for ind in offspring]
        
        combined_pop = pop_ext + offspring
        combined_scores = scores_ext + new_scores
        pop_ext, scores_ext = nsga2_select(combined_pop, combined_scores, pop_size)
        pbar.set_postfix(cache_size=len(fitness_cache_ext))

    fronts_ext = fast_non_dominated_sort(pop_ext, scores_ext)
    pareto_ext = []
    seen = set()
    for i in fronts_ext[0]:
        key = tuple(pop_ext[i])
        if key in seen: continue
        seen.add(key)
        astr, C, nc = decode_ext(pop_ext[i])
        score, params = scores_ext[i]
        
        match = df_base[df_base['arch_str'] == astr]
        acc_bench = float(match['acc_valid'].values[0]) if (len(match) > 0 and C==16 and nc==5) else None
        
        pareto_ext.append({
            'arch_str' : astr,
            'channel'  : C,
            'num_cells': nc,
            'proxy_score': score,
            'params_k' : params,
            'acc_bench': acc_bench,
            'in_benchmark': (len(match)>0 and C==16 and nc==5),
        })

    return pd.DataFrame(pareto_ext).sort_values('params_k')
