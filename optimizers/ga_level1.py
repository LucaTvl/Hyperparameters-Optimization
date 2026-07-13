import random
from config.settings import ZCP_COLS
from optimizers.genetic_ops import random_weights, tournament, blx_crossover, mutate_gaussian
try:
    from tqdm.auto import tqdm
except ImportError:
    from tqdm import tqdm

def weights_to_dict(w, cols=ZCP_COLS):
    return {col: w[i] for i, col in enumerate(cols)}

def precision_at_k(weights: dict, df_eval, k: int = 50, cols=ZCP_COLS):
    """
    Calcula el Precision@K del proxy generado por la combinación lineal de los pesos.
    """
    score = sum(weights[col] * df_eval[f'{col}_norm'] for col in cols)
    rank_proxy = score.rank(ascending=False, method='min')
    top_real = set(df_eval.nsmallest(k, 'rank_real')['arch_id'])
    top_proxy = set(df_eval[rank_proxy <= k]['arch_id'])
    return len(top_real & top_proxy) / k

def run_ga_level1(df_calib, pop_size=20, n_gen=5, p_cross=0.8, p_mut=0.3, elite_k=2, sigma=0.15, k_precision=50, seed=42, cols=ZCP_COLS):
    """
    GA Nivel 1: Optimiza el vector de pesos del proxy en el espacio continuo.
    """
    random.seed(seed)
    n_weights = len(cols)
    pop = [random_weights(n_weights) for _ in range(pop_size)]
    
    def fitness(w):
        return precision_at_k(weights_to_dict(w, cols), df_calib, k=k_precision, cols=cols)
    
    fits = [fitness(ind) for ind in pop]
    history = [max(fits)]

    pbar = tqdm(range(n_gen), desc="GA Nivel 1", leave=False)
    for gen in pbar:
        sorted_idx = sorted(range(len(pop)), key=lambda i: fits[i], reverse=True)
        new_pop = [pop[i][:] for i in sorted_idx[:elite_k]]

        while len(new_pop) < pop_size:
            p1 = tournament(pop, fits)
            p2 = tournament(pop, fits)
            child = blx_crossover(p1, p2) if random.random() < p_cross else p1[:]
            if random.random() < p_mut:
                child = mutate_gaussian(child, sigma)
            new_pop.append(child)

        pop = new_pop
        fits = [fitness(ind) for ind in pop]
        best_f = max(fits)
        history.append(best_f)
        pbar.set_postfix(best_p50=f"{best_f:.4f}")

    best_idx = max(range(len(pop)), key=lambda i: fits[i])
    return pop[best_idx], fits[best_idx], history
