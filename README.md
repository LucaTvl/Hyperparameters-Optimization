# Hyperparameters Optimization & Neural Architecture Search (NAS)

Este proyecto implementa un framework de optimización de hiperparámetros y búsqueda de arquitectura neuronal (NAS) utilizando **Zero-Cost Proxies (ZCP)**, algoritmos genéticos (**GA**) y optimización multiobjetivo (**NSGA-II**) sobre el espacio de búsqueda de NATS-Bench y su espacio extendido, entrenando y evaluando modelos en **CIFAR-10**.

---

## Manejo de Datos y Archivos CSV (Primer Uso o Sin CSVs Previos)

El pipeline experimental utiliza archivos tabulares en la carpeta `CSV/` para almacenar y consultar de forma instantánea las evaluaciones de las arquitecturas y los proxies:

| Archivo CSV | Descripción | Contenido |
| :--- | :--- | :--- |
| `nats_ground_truth.csv` | Ground Truth de NATS-Bench | Métricas reales (accuracy de validación y test a 200 épocas, parámetros, flops, latencia) para las 15.625 arquitecturas. |
| `nats_zcps.csv` | Muestra de calibración de ZCPs | Muestra estratificada (~1.001 arquitecturas) utilizada por el Algoritmo Genético Nivel 1 (GA-1) para calibrar los pesos de los proxies. |
| `nats_zcps_full.csv` | ZCPs del espacio base completo | Zero-Cost Proxies calculados para las 15.625 arquitecturas del espacio tabular. |
| `nats_zcps_extended.csv` | ZCPs del espacio extendido | Proxies evaluados variando cantidad de canales ($C \in \{8, 16, 32, 64\}$) y celdas ($N \in \{3, 5, 7, 9\}$). |

### ¿Qué ocurre si no tienes los archivos CSV?

El framework implementa **detección condicional y generación manual automática**:

1. **Detección Automática (`if os.path.exists`)**:
   Al iniciar `main.py`, el sistema verifica si los archivos requeridos están presentes en la carpeta `CSV/`:
   - Si los detecta, los carga inmediatamente en disco en cuestión de segundos.
   - Si **no los detecta**, activa automáticamente la rama de **carga manual / cálculo desde cero**.

2. **Requisitos para la Generación Manual**:
   - **NATS-Bench TSS**: Para construir `nats_ground_truth.csv`, se requiere descargar el archivo del benchmark [NATS-Bench](https://github.com/D-X-Y/NATS-Bench) (`NATS-tss-v1_0-3ffb9-simple.tar`) y descomprimirlo en la ruta configurada en `config/settings.py` (por defecto: `CSV/NATS-tss-v1_0-3ffb9-simple/NATS-tss-v1_0-3ffb9-simple/`).
   - **Dataset CIFAR-10**: Para el cálculo de los Zero-Cost Proxies, `torchvision` descargará automáticamente CIFAR-10 en `tmp/cifar10/` (si no existe) para extraer los mini-batches requeridos por las métricas basadas en gradientes.
   - **Dispositivo**: Se recomienda ejecutar con GPU (CUDA) para que el cómputo de proxies sea rápido.

3. **Tiempos Estimados de Generación**:
   - **Ground Truth (`nats_ground_truth.csv`)**: ~10–15 minutos (se extraen las 15.625 arquitecturas en paralelo mediante `ThreadPoolExecutor`).
   - **ZCPs de Calibración (`nats_zcps.csv`)**: ~1–3 minutos en GPU.
   - **ZCPs Completos (`nats_zcps_full.csv`)**: Reutiliza las arquitecturas ya calculadas en la calibración y computa las restantes secuencialmente en GPU.

> Una vez generados los CSVs, quedan guardados permanentemente en la carpeta `CSV/`. En todas las ejecuciones posteriores, el sistema los detectará de inmediato y no volverá a calcularlos.

---

## Estructura de Scripts Principales

A continuación se detalla el propósito y uso de los scripts principales del repositorio: `main.py` y `demo_predict.py`.

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