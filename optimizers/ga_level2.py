import random
import numpy as np
from config.settings import OPS_LIST
from optimizers.genetic_ops import random_ind, one_point_crossover, mutate
from optimizers.ag_core import IndividuoAG, PoblacionAG, TorneoAG, ElitismoAG, OperadoresAG, AlgoritmoGeneticoAG
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
    GA Nivel 2: Algoritmo Genético Discreto para búsqueda de arquitecturas neuronales.
    
    Conceptos Evolutivos:
    - Individuo: Una arquitectura neuronal candidata.
    - Cromosoma: Lista de 6 genes categóricos/enteros. Cada gen representa una operación en la celda.
    - Fitness: Puntuación del proxy calibrado. (No es la accuracy real, sino una estimación rápida).
    - Accuracy Real: Se usa solo para registrar o comparar resultados al final, no guía la evolución.
    - Selección: Torneo determinístico.
    - Cruce: De un punto (One-Point Crossover).
    - Mutación: Aleatoria discreta (altera un gen por otra operación permitida).
    - Elitismo: Se conservan los 'elite_k' mejores individuos.
    - Caché: Se almacenan las evaluaciones previas para no recalcular arquitecturas ya vistas.
    """
    random.seed(seed)
    
    # 1. Inicialización de la población
    individuos_iniciales = [IndividuoAG(random_ind()) for _ in range(pop_size)]
    poblacion = PoblacionAG(individuos_iniciales)
    
    # 2. Evaluación inicial usando el caché
    cache = {}
    evals_tracker = {'n_evals': 0}
    
    def fitness_fn(genes):
        k = tuple(genes)
        if k not in cache:
            cache[k] = get_proxy_score(genes, proxy_lookup)
            evals_tracker['n_evals'] += 1
        return cache[k]
        
    seleccion = TorneoAG(k=3)
    elitismo = ElitismoAG(cantidad_elite=elite_k)
    
    def mutacion_fn(g):
        return mutate(g, p_mut)
        
    def cruce_fn(g1, g2):
        if random.random() < p_cross:
            return one_point_crossover(g1, g2)
        return (g1[:], g2[:])
        
    operadores = OperadoresAG(fn_cruce=cruce_fn, fn_mutacion=mutacion_fn)
    
    ag = AlgoritmoGeneticoAG(poblacion, seleccion, operadores, elitismo, fitness_fn)
    
    history_compat = []
    
    def callback(gen, pob, mets):
        # En el codigo viejo, gen 0 era la primera evaluacion post-generacion.
        # Aqui callback(-1) es pre-generacion. Lo ignoramos para mantener compatibilidad 
        # o lo ajustamos si es necesario. El viejo history trackeaba durante el bucle.
        if gen == -1: return
        
        best_f = mets['mejor_fitness']
        mejores = pob.obtener_mejores(1)
        astr = genes_to_arch_str(mejores[0].genes)
        info = proxy_lookup.get(astr, {})
        history_compat.append({
            'gen'      : gen,
            'best_fit' : best_f,
            'mean_fit' : mets['promedio'],
            'n_evals'  : evals_tracker['n_evals'],
            'rank_real': info.get('rank_real', np.nan),
            'acc_valid': info.get('acc_valid', np.nan),
        })

    pbar = tqdm(total=n_gen, desc=f"GA Nivel 2 (seed={seed})", leave=False)
    
    def callback_pbar(gen, pob, mets):
        callback(gen, pob, mets)
        pbar.set_postfix(best_fit=f"{mets['mejor_fitness']:.4f}")

    poblacion_final = ag.ejecutar(n_gen, pbar=pbar, callback=callback_pbar)
    pbar.close()

    mejores = poblacion_final.obtener_mejores(1)
    best_ind = mejores[0].genes
    best_fit = mejores[0].fitness
    best_astr = genes_to_arch_str(best_ind)
    best_info = proxy_lookup.get(best_astr, {})
    
    return best_ind, best_fit, history_compat, best_info, evals_tracker['n_evals']

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
