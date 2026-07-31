import random
from config.settings import ZCP_COLS
from optimizers.genetic_ops import random_weights, blx_crossover, mutate_gaussian
from optimizers.ag_core import IndividuoAG, PoblacionAG, TorneoAG, ElitismoAG, OperadoresAG, AlgoritmoGeneticoAG
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
    GA Nivel 1: Algoritmo Genético Continuo para calibrar pesos reales de los estimadores sin entrenamiento.
    
    Conceptos Evolutivos:
    - Individuo: Vector de pesos reales.
    - Cromosoma: Lista de genes continuos en el rango [-1.0, 1.0]. Cada gen es el peso de un estimador.
    - Fitness: Métrica Precision@K (ej. Precision@50) que evalúa el proxy combinado.
    - Selección: Torneo determinístico.
    - Cruce: BLX-alpha (adecuado para espacios continuos).
    - Mutación: Gaussiana adaptativa.
    - Elitismo: Se conservan los 'elite_k' mejores individuos.
    
    El resultado final son los pesos del proxy combinado optimizados.
    """
    random.seed(seed)
    n_weights = len(cols)
    
    # 1. Inicialización de la población
    individuos_iniciales = [IndividuoAG(random_weights(n_weights)) for _ in range(pop_size)]
    poblacion = PoblacionAG(individuos_iniciales)
    
    def fitness_fn(w):
        return precision_at_k(weights_to_dict(w, cols), df_calib, k=k_precision, cols=cols)
        
    seleccion = TorneoAG(k=3)
    elitismo = ElitismoAG(cantidad_elite=elite_k)
    
    # En GA 1, la mutacion gaussiana toma el ind y sigma. 
    def mutacion_fn(g):
        if random.random() < p_mut:
            return mutate_gaussian(g, sigma)
        return g[:]
        
    def cruce_fn(g1, g2):
        if random.random() < p_cross:
            return blx_crossover(g1, g2)
        return g1[:]
        
    operadores = OperadoresAG(fn_cruce=cruce_fn, fn_mutacion=mutacion_fn)
    
    ag = AlgoritmoGeneticoAG(poblacion, seleccion, operadores, elitismo, fitness_fn)
    
    # Para mantener compatibilidad con history que era [max_fit_gen0, max_fit_gen1...]
    history_compat = []
    def callback(gen, pob, mets):
        # Se incluye gen = -1 (evaluación inicial antes de la primera generación)
        # porque el código anterior registraba la población inicial en history
        history_compat.append(mets['mejor_fitness'])

    pbar = tqdm(total=n_gen, desc="GA Nivel 1", leave=False)
    poblacion_final = ag.ejecutar(n_gen, pbar=pbar, callback=callback)
    pbar.close()

    # Reconstruir formato de salida para mantener compatibilidad
    mejores = poblacion_final.obtener_mejores(1)
    best_ind = mejores[0].genes
    best_fit = mejores[0].fitness
    
    return best_ind, best_fit, history_compat
