import random
import numpy as np
from config.settings import OPS_LIST
from optimizers.genetic_ops import random_ind, tournament, one_point_crossover, mutate
try:
    from tqdm.auto import tqdm
except ImportError:
    from tqdm import tqdm

def build_proxy(weights: dict, df_eval, cols):
    """Construye la puntuación proxy ponderando los ZCPs normalizados."""
    return sum(weights[col] * df_eval[f'{col}_norm'] for col in cols)

def genes_to_arch_str(genes):
    """Decodifica un cromosoma en el string de arquitectura (NATS-Bench base)."""
    ops = [OPS_LIST[g] for g in genes]
    return (f'|{ops[0]}~0|+|{ops[1]}~0|{ops[2]}~1|'
            f'+|{ops[3]}~0|{ops[4]}~1|{ops[5]}~2|')

def greedy_search(df_full, k_consult: int):
    """Toma las top-k arquitecturas según el proxy y devuelve la mejor real."""
    top_k = df_full.nlargest(k_consult, 'proxy_ga')
    best = top_k.loc[top_k['acc_valid'].idxmax()]
    return {
        'acc_valid': best['acc_valid'],
        'rank_real': int(best['rank_real']),
        'params_k' : best['params_k'],
        'arch_str' : best['arch_str']
    }

def get_proxy_score(genes, proxy_lookup):
    astr = genes_to_arch_str(genes)
    return proxy_lookup.get(astr, {}).get('proxy_ga', -np.inf)

def run_ga_level2(proxy_lookup, pop_size=50, n_gen=30, p_mut=0.2, p_cross=0.8, elite_k=3, seed=42):
    """
    GA Nivel 2: Búsqueda de arquitectura discreta utilizando el proxy como fitness.
    """
    random.seed(seed)
    pop = [random_ind() for _ in range(pop_size)]
    fits = [get_proxy_score(ind, proxy_lookup) for ind in pop]
    cache = {tuple(ind): f for ind, f in zip(pop, fits)}

    history = []
    n_evals = pop_size

    pbar = tqdm(range(n_gen), desc=f"GA Nivel 2 (seed={seed})", leave=False)
    for gen in pbar:
        best_f = max(fits)
        best_i = np.argmax(fits)
        astr = genes_to_arch_str(pop[best_i])
        info = proxy_lookup.get(astr, {})
        history.append({
            'gen'      : gen,
            'best_fit' : best_f,
            'mean_fit' : np.mean(fits),
            'n_evals'  : n_evals,
            'rank_real': info.get('rank_real', np.nan),
            'acc_valid': info.get('acc_valid', np.nan),
        })

        sorted_idx = np.argsort(fits)[::-1]
        new_pop = [pop[i][:] for i in sorted_idx[:elite_k]]

        while len(new_pop) < pop_size:
            p1 = tournament(pop, fits)
            p2 = tournament(pop, fits)
            c1, c2 = one_point_crossover(p1, p2) if random.random() < p_cross else (p1[:], p2[:])
            new_pop.append(mutate(c1, p_mut))
            if len(new_pop) < pop_size:
                new_pop.append(mutate(c2, p_mut))

        pop = new_pop
        new_fits = []
        for ind in pop:
            k = tuple(ind)
            if k not in cache:
                cache[k] = get_proxy_score(ind, proxy_lookup)
                n_evals += 1
            new_fits.append(cache[k])
        fits = new_fits
        pbar.set_postfix(best_fit=f"{max(fits):.4f}")

    best_idx = np.argmax(fits)
    best_astr = genes_to_arch_str(pop[best_idx])
    best_info = proxy_lookup.get(best_astr, {})
    return pop[best_idx], fits[best_idx], history, best_info, n_evals

def run_rs_level2(proxy_lookup, n_evals=1500, seed=42):
    random.seed(seed)
    best_f, best_info = -np.inf, {}
    history = []
    for i in range(n_evals):
        ind = random_ind()
        f = get_proxy_score(ind, proxy_lookup)
        if f > best_f:
            best_f = f
            astr = genes_to_arch_str(ind)
            best_info = proxy_lookup.get(astr, {})
        if (i+1) % 50 == 0:
            history.append({'n_evals': i+1, 'best_fit': best_f,
                            'rank_real': best_info.get('rank_real', np.nan),
                            'acc_valid': best_info.get('acc_valid', np.nan)})
    return best_f, best_info, history
