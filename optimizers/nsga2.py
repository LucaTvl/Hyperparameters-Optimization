import random
import numpy as np
import pandas as pd
from optimizers.genetic_ops import random_ind, one_point_crossover, mutate
from optimizers.ga_level2 import genes_to_arch_str
try:
    from tqdm.auto import tqdm
except ImportError:
    from tqdm import tqdm

def dominates(a, b):
    """
    Verifica si la solución 'a' domina según Pareto a la solución 'b'.
    Asume que el Objetivo 1 (a[0]) se debe MAXIMIZAR y el Objetivo 2 (a[1]) se debe MINIMIZAR.
    """
    return (a[0] >= b[0] and a[1] <= b[1] and (a[0] > b[0] or a[1] < b[1]))

def fast_non_dominated_sort(pop, scores):
    n = len(pop)
    dominated_by = [0] * n
    dominates_list = [[] for _ in range(n)]
    fronts = [[]]

    for i in range(n):
        for j in range(n):
            if i == j: continue
            if dominates(scores[i], scores[j]):
                dominates_list[i].append(j)
            elif dominates(scores[j], scores[i]):
                dominated_by[i] += 1
        if dominated_by[i] == 0:
            fronts[0].append(i)

    k = 0
    while fronts[k]:
        next_front = []
        for i in fronts[k]:
            for j in dominates_list[i]:
                dominated_by[j] -= 1
                if dominated_by[j] == 0:
                    next_front.append(j)
        k += 1
        fronts.append(next_front)
    return fronts[:-1]

def crowding_distance(scores, front):
    """
    Calcula la distancia de hacinamiento (Crowding Distance) para mantener diversidad.
    Asigna np.inf a las soluciones extremas para preservarlas siempre.
    """
    if len(front) <= 2:
        return [np.inf] * len(front)
    dists = [0.0] * len(front)
    for obj in range(2):
        vals = [scores[i][obj] for i in front]
        order = np.argsort(vals)
        dists[order[0]] = dists[order[-1]] = np.inf
        v_range = vals[order[-1]] - vals[order[0]] + 1e-10
        for k in range(1, len(front)-1):
            dists[order[k]] += (vals[order[k+1]] - vals[order[k-1]]) / v_range
    return dists

def nsga2_select(pop, scores, pop_size):
    fronts = fast_non_dominated_sort(pop, scores)
    new_pop, new_scores = [], []
    for front in fronts:
        if len(new_pop) + len(front) <= pop_size:
            for i in front:
                new_pop.append(pop[i][:])
                new_scores.append(scores[i])
        else:
            cd = crowding_distance(scores, front)
            sorted_front = sorted(range(len(front)), key=lambda x: cd[x], reverse=True)
            needed = pop_size - len(new_pop)
            for idx in sorted_front[:needed]:
                new_pop.append(pop[front[idx]][:])
                new_scores.append(scores[front[idx]])
            break
    return new_pop, new_scores

def run_nsga2(proxy_lookup, pop_size=80, n_gen=40, p_mut=0.2, p_cross=0.8, seed=42):
    """
    NSGA-II: Algoritmo Evolutivo Multiobjetivo.
    No es un GA clásico, sino que busca un conjunto de soluciones (Frontera de Pareto).
    
    Conceptos Evolutivos:
    - Individuo: Una arquitectura neuronal candidata (cromosoma discreto de 6 genes).
    - Población: Conjunto de individuos evolucionando simultáneamente.
    - Objetivo 1: Maximizar el proxy calibrado ('proxy_ga').
    - Objetivo 2: Minimizar la complejidad, medida en cantidad de parámetros ('params_k').
    - Dominancia de Pareto: Una solución es mejor si mejora en un objetivo sin empeorar el otro.
    - Frentes no dominados: Clasificación de la población según niveles de dominancia (rankings).
    - Distancia de hacinamiento: Métrica para favorecer zonas menos exploradas del frente de Pareto.
    - Selección: Torneo binario basado en el ranking del frente y, en caso de empate, en la crowding distance.
    """
    random.seed(seed); np.random.seed(seed)
    pop = [random_ind() for _ in range(pop_size)]
    
    def get_objectives(genes):
        astr = genes_to_arch_str(genes)
        info = proxy_lookup.get(astr, {})
        return (info.get('proxy_ga', -np.inf), info.get('params_k', np.inf))
        
    scores = [get_objectives(ind) for ind in pop]
    n_evals = pop_size
    history_fronts = []

    pbar = tqdm(range(n_gen), desc=f"NSGA-II (seed={seed})", leave=False)
    for gen in pbar:
        offspring = []
        while len(offspring) < pop_size:
            fronts = fast_non_dominated_sort(pop, scores)
            ranks = [0]*len(pop)
            for r, front in enumerate(fronts):
                for i in front: ranks[i] = r

            def nsga_tournament(k=2):
                cands = random.sample(range(len(pop)), k)
                c = min(cands, key=lambda i: (
                    ranks[i],
                    -crowding_distance(scores, fronts[ranks[i]])[fronts[ranks[i]].index(i) if i in fronts[ranks[i]] else 0]
                ))
                return pop[c][:]

            p1, p2 = nsga_tournament(), nsga_tournament()
            c1, c2 = one_point_crossover(p1,p2) if random.random()<p_cross else (p1[:],p2[:])
            offspring.append(mutate(c1, p_mut))
            if len(offspring) < pop_size:
                offspring.append(mutate(c2, p_mut))
            n_evals += 1

        combined_pop = pop + offspring
        combined_scores = scores + [get_objectives(ind) for ind in offspring]
        pop, scores = nsga2_select(combined_pop, combined_scores, pop_size)

        fronts = fast_non_dominated_sort(pop, scores)
        pf = [(scores[i][0], scores[i][1]) for i in fronts[0]]
        history_fronts.append(pf)
        pbar.set_postfix(pareto_size=len(fronts[0]))

    fronts = fast_non_dominated_sort(pop, scores)
    pareto = []
    for i in fronts[0]:
        astr = genes_to_arch_str(pop[i])
        info = proxy_lookup.get(astr, {})
        pareto.append({
            'arch_str' : astr,
            'proxy_ga' : scores[i][0],
            'params_k' : scores[i][1],
            'acc_valid': info.get('acc_valid', np.nan),
            'rank_real': info.get('rank_real', np.nan),
        })
    return pd.DataFrame(pareto), history_fronts, n_evals
