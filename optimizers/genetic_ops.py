import random
from config.settings import N_EDGES

W_MIN, W_MAX = -1.0, 1.0

# ── Continuos (GA Nivel 1) ──

def random_weights(n_weights):
    return [random.uniform(W_MIN, W_MAX) for _ in range(n_weights)]

def blx_crossover(p1, p2, alpha=0.3):
    """BLX-α: Genera hijos en el intervalo extendido entre padres."""
    child = []
    for a, b in zip(p1, p2):
        lo, hi = min(a,b), max(a,b)
        d = hi - lo
        lo_ext = max(W_MIN, lo - alpha*d)
        hi_ext = min(W_MAX, hi + alpha*d)
        child.append(random.uniform(lo_ext, hi_ext))
    return child

def mutate_gaussian(ind, sigma=0.15):
    """Mutación gaussiana adaptativa."""
    return [max(W_MIN, min(W_MAX, g + random.gauss(0, sigma))) for g in ind]

# ── Discretos (GA Nivel 2 y NSGA-II) ──

def random_ind():
    """Individuo para el espacio base NATS-Bench."""
    return [random.randint(0,4) for _ in range(N_EDGES)]

def one_point_crossover(p1, p2):
    pt = random.randint(1, len(p1)-1)
    return p1[:pt]+p2[pt:], p2[:pt]+p1[pt:]

def mutate(ind, p_mut=0.2):
    return [random.randint(0,4) if random.random() < p_mut else g for g in ind]

def random_ind_ext():
    """Individuo para el espacio extendido (8 genes)."""
    return [random.randint(0,4) for _ in range(6)] + [random.randint(0,3), random.randint(0,3)]

def mutate_ext(ind, p_mut=0.15):
    new = ind[:]
    for i in range(6):
        if random.random() < p_mut:
            new[i] = random.randint(0, 4)
    if random.random() < p_mut: new[6] = random.randint(0, 3)
    if random.random() < p_mut: new[7] = random.randint(0, 3)
    return new

def crossover_ext(p1, p2):
    pt = random.randint(1, 7)
    return p1[:pt]+p2[pt:], p2[:pt]+p1[pt:]

# ── Compartidos ──

def tournament(pop, fits, k=3):
    """Torneo determinístico de tamaño k."""
    idx = random.sample(range(len(pop)), k)
    return pop[max(idx, key=lambda i: fits[i])][:]
