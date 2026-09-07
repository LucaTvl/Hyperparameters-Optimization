# Hyperparameters Optimization & Neural Architecture Search (NAS)

Este proyecto implementa un framework de optimización de hiperparámetros y búsqueda de arquitectura neuronal (NAS) utilizando **Zero-Cost Proxies (ZCP)**, algoritmos genéticos (**GA**) y optimización multiobjetivo (**NSGA-II**) sobre el espacio de búsqueda de NATS-Bench y su espacio extendido, entrenando y evaluando modelos en **CIFAR-10**.

---

## Estructura de Scripts Principales

A continuación se detalla el propósito y uso de los tres scripts principales del repositorio: `main.py`, `demo_predict.py` y `predict_image.py`.

---

### 1. `main.py` — Pipeline Completo de Búsqueda, Optimización y Entrenamiento

Es el script central del proyecto. Ejecuta todo el flujo experimental automatizado:
1. **Zero-Cost Proxies (ZCP)**: Carga o calcula las métricas de costo cero (SNIP, Grasp, SynFlow, Jacob Cov, Grad Norm) sobre el espacio de arquitecturas.
2. **Algoritmo Genético Nivel 1 (GA-1)**: Optimiza los pesos ponderados de los proxies para maximizar la correlación (`Precision@50`) con la precisión real.
3. **Búsqueda en Espacio Extendido (Nivel 2)**:
   - Explora variaciones de topología, canales ($C \in \{8, 16, 32, 64\}$) y cantidad de celdas ($N \in \{3, 5, 7, 9\}$).
   - Compara **GA Nivel 2**, **Random Search (RS)** y **Greedy Search**.
4. **Optimización Multiobjetivo (NSGA-II)**: Obtiene el frente de Pareto balanceando rendimiento (proxy score) y complejidad del modelo (número de parámetros).
5. **Entrenamiento Completo y Exportación**:
   - Entrena desde cero las mejores arquitecturas seleccionadas durante 200 épocas en CIFAR-10.
   - Guarda los modelos entrenados en `outputs/modelos_<timestamp>/` en formato `.pt`.
   - Genera gráficos de convergencia, frentes de Pareto y métricas de rendimiento en `outputs/graficos_<timestamp>/`.

#### Ejecución:
```bash
python main.py
```
> **Nota**: Utiliza aceleración por GPU (CUDA) automáticamente si se encuentra disponible.

---

### 2. `demo_predict.py` — Evaluación y Pruebas del Test Set de CIFAR-10

Diseñado para auditar, probar y evaluar exhaustivamente modelos ya entrenados (`.pt`) frente al conjunto de prueba de CIFAR-10 o imágenes de muestra.

#### Parámetros principales:
- `--model <ruta.pt>` *(Requerido)*: Ruta al checkpoint del modelo entrenado.
- Modo de selección (mutuamente excluyentes, elegir uno):
  - `--idx <int>`: Clasifica una imagen específica del test set de CIFAR-10 por su índice (0 a 9999).
  - `--random`: Elige y clasifica una imagen aleatoria del test set de CIFAR-10.
  - `--image <ruta>`: Clasifica una imagen externa (.jpg, .png, etc.).
  - `--eval-n <N>`: Evalúa la precisión (`Top-1` y `Top-3`) sobre una muestra aleatoria de $N$ imágenes.
  - `--eval-all`: Evalúa el modelo en las 10.000 imágenes completas del test set.
- Parámetros adicionales:
  - `--batch-size <int>`: Tamaño de batch para evaluación masiva (default: `256`).
  - `--eval-seed <int>`: Semilla para reproducibilidad en `--eval-n` (default: `123`).

#### Ejemplos de uso:

- **Evaluar una imagen aleatoria del test set:**
  ```bash
  python demo_predict.py --model outputs/modelos/C64_nc9_acc94.33.pt --random
  ```

- **Evaluar una imagen por índice específico:**
  ```bash
  python demo_predict.py --model outputs/modelos/C64_nc9_acc94.33.pt --idx 42
  ```

- **Evaluar la precisión sobre 1000 imágenes del test set:**
  ```bash
  python demo_predict.py --model outputs/modelos/C64_nc9_acc94.33.pt --eval-n 1000
  ```

- **Evaluar todo el test set completo (10.000 imágenes):**
  ```bash
  python demo_predict.py --model outputs/modelos/C64_nc9_acc94.33.pt --eval-all
  ```
  *(Guarda un informe detallado con accuracy por clase en `predictions/<nombre_modelo>/eval_all_<timestamp>.txt`)*.

---

## Clases Reconocidas (CIFAR-10)

Los modelos entrenados identifican 10 categorías:
1. **avión** (*airplane*)
2. **auto** (*automobile*)
3. **pájaro** (*bird*)
4. **gato** (*cat*)
5. **ciervo** (*deer*)
6. **perro** (*dog*)
7. **rana** (*frog*)
8. **caballo** (*horse*)
9. **barco** (*ship*)
10. **camión** (*truck*)