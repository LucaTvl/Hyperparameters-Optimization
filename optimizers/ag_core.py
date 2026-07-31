import random
import copy

class IndividuoAG:
    """
    Representa un individuo del algoritmo genético.
    En GA Nivel 1, sus genes son pesos reales.
    En GA Nivel 2, sus genes son operaciones de una arquitectura.
    """
    def __init__(self, genes):
        self.genes = genes
        self.fitness = None

    def clone(self):
        """Devuelve una copia profunda segura del individuo."""
        ind = IndividuoAG(copy.deepcopy(self.genes))
        ind.fitness = self.fitness
        return ind

class PoblacionAG:
    """
    Administra una población de IndividuoAG.
    Permite evaluar fitness, obtener mejores individuos y calcular métricas.
    """
    def __init__(self, individuos):
        self.individuos = individuos

    def evaluar(self, fitness_fn):
        """Evalúa los individuos que aún no tienen fitness."""
        for ind in self.individuos:
            if ind.fitness is None:
                ind.fitness = fitness_fn(ind.genes)

    def obtener_mejores(self, n):
        """Devuelve los n mejores individuos (mayor fitness)."""
        return sorted(self.individuos, key=lambda x: x.fitness, reverse=True)[:n]

    def metricas(self):
        """Devuelve métricas de la población."""
        fits = [ind.fitness for ind in self.individuos if ind.fitness is not None]
        if not fits:
            return {}
        
        mejores = self.obtener_mejores(1)
        mejor_ind = mejores[0] if mejores else None
        
        promedio = sum(fits) / len(fits)
        desviacion = (sum((f - promedio) ** 2 for f in fits) / len(fits)) ** 0.5
        return {
            'mejor_fitness': max(fits),
            'peor_fitness': min(fits),
            'promedio': promedio,
            'desviacion': desviacion,
            'mejor_genes': mejor_ind.genes if mejor_ind else None,
            'mejor_individuo': mejor_ind
        }

class SeleccionAG:
    """Clase base para métodos de selección."""
    def seleccionar(self, poblacion):
        raise NotImplementedError

class TorneoAG(SeleccionAG):
    """Selección por torneo."""
    def __init__(self, k=3):
        self.k = k

    def seleccionar(self, poblacion):
        k_real = min(int(self.k), len(poblacion.individuos))
        cands = random.sample(poblacion.individuos, k_real)
        return max(cands, key=lambda x: x.fitness)

class RuletaAG(SeleccionAG):
    """
    Selección por ruleta.
    Maneja valores negativos desplazando el mínimo a cero (shift).
    (Implementada por completitud conceptual, se recomienda Torneo para NAS).
    """
    def seleccionar(self, poblacion):
        fits = [ind.fitness for ind in poblacion.individuos]
        min_f = min(fits)
        offset = 0 if min_f >= 0 else -min_f + 1e-6
        shifted = [f + offset for f in fits]
        
        total = sum(shifted)
        if total == 0:
            return random.choice(poblacion.individuos)
            
        r = random.uniform(0, total)
        acum = 0
        for ind, f_shift in zip(poblacion.individuos, shifted):
            acum += f_shift
            if acum >= r:
                return ind
        return poblacion.individuos[-1]

class ElitismoAG:
    """Conserva los mejores individuos de la población sin modificarlos."""
    def __init__(self, cantidad_elite):
        self.cantidad_elite = cantidad_elite

    def obtener_elite(self, poblacion):
        """Retorna clones seguros de la élite para que no muten accidentalmente."""
        mejores = poblacion.obtener_mejores(self.cantidad_elite)
        return [ind.clone() for ind in mejores]

class OperadoresAG:
    """Aplica cruce y mutación usando funciones específicas del dominio."""
    def __init__(self, fn_cruce, fn_mutacion):
        self.fn_cruce = fn_cruce
        self.fn_mutacion = fn_mutacion

    def aplicar(self, p1, p2):
        """
        Aplica cruce y mutación.
        La lógica de probabilidad (si se cruzan o no, y si mutan o no)
        queda delegada a las funciones inyectadas (fn_cruce, fn_mutacion).
        """
        hijos_genes = self.fn_cruce(p1.genes, p2.genes)
        if not isinstance(hijos_genes, tuple):
            hijos_genes = (hijos_genes,)

        nuevos_individuos = []
        for g in hijos_genes:
            g_mut = self.fn_mutacion(g)
            nuevos_individuos.append(IndividuoAG(g_mut))
            
        return nuevos_individuos

class AlgoritmoGeneticoAG:
    """
    Orquesta el ciclo evolutivo:
    inicialización -> evaluación -> selección -> cruce -> mutación -> elitismo -> reemplazo -> registro.
    """
    def __init__(self, poblacion, seleccion, operadores, elitismo, fitness_fn):
        self.poblacion = poblacion
        self.seleccion = seleccion
        self.operadores = operadores
        self.elitismo = elitismo
        self.fitness_fn = fitness_fn
        self.historial = []

    def ejecutar(self, n_gen, pbar=None, callback=None):
        """
        Ejecuta n generaciones. 
        callback: función(gen, poblacion, metricas) para registrar información adicional a medida que evoluciona.
        """
        # Evaluación inicial
        self.poblacion.evaluar(self.fitness_fn)
        
        # Registro pre-generacional
        if callback:
            callback(-1, self.poblacion, self.poblacion.metricas())
            
        for gen in range(n_gen):
            n_poblacion = len(self.poblacion.individuos)
            nueva_poblacion = self.elitismo.obtener_elite(self.poblacion)
            
            while len(nueva_poblacion) < n_poblacion:
                padre1 = self.seleccion.seleccionar(self.poblacion)
                padre2 = self.seleccion.seleccionar(self.poblacion)
                hijos = self.operadores.aplicar(padre1, padre2)
                
                for hijo in hijos:
                    if len(nueva_poblacion) < n_poblacion:
                        nueva_poblacion.append(hijo)
            
            self.poblacion = PoblacionAG(nueva_poblacion)
            self.poblacion.evaluar(self.fitness_fn)
            
            mets = self.poblacion.metricas()
            self.historial.append(mets)
            
            if callback:
                callback(gen, self.poblacion, mets)
                
            if pbar:
                pbar.update(1)
                
        return self.poblacion
