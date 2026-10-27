import datetime
import platform
import sys
import torch
import numpy as np

def write_initial_metadata(metadata_file, script_path, git_commit, device, dataset, cifar_dir, csv_gt, csv_zcp, graficos_dir, master_seed):
    metadata_initial = f"""============================================================
METADATA DE LA EJECUCIÓN
============================================================

Fecha y hora de inicio: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Archivo ejecutado: {script_path}
Commit Git: {git_commit}
Sistema operativo: {platform.system()} {platform.release()}
Versión de Python: {sys.version.split()[0]}
Versión de PyTorch: {torch.__version__}
CUDA disponible: {torch.cuda.is_available()}
Versión CUDA informada por PyTorch: {torch.version.cuda if torch.cuda.is_available() else 'No disponible'}
GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No disponible'}
Dispositivo utilizado: {device}

Dataset: {dataset}
Ruta del dataset: {cifar_dir}
Partición metodológica:
  - Entrenamiento: 45.000 imágenes (con RandomCrop y RandomHorizontalFlip)
  - Validación: 5.000 imágenes (sin aumentación, para selección y mejor checkpoint)
  - Test: 10.000 imágenes (evaluación única final con mejor checkpoint)
Archivo de ground truth: {csv_gt}
Archivo de Zero-Cost Proxies: {csv_zcp}
Carpeta de gráficos: {graficos_dir}

------------------------------------------------------------
SEEDS EXISTENTES
------------------------------------------------------------

MASTER_SEED: {master_seed}

Seeds derivadas (todas reproducibles a partir de MASTER_SEED):
  GA Nivel 1 (run 0-9): MASTER_SEED + run*17
  RS Nivel 1 (run 0-9): MASTER_SEED + run*31 + 100
  ZCPs espacio extendido: MASTER_SEED + 300
  GA Nivel 2 (3 corridas): MASTER_SEED + 400, +407, +499
  RS Nivel 2: MASTER_SEED + 500
  NSGA-II benchmark: MASTER_SEED + 600
  RS multi-objetivo: MASTER_SEED + 600
  NSGA-II extendido: MASTER_SEED + 700

------------------------------------------------------------
CONFIGURACIÓN
------------------------------------------------------------

GA Nivel 1:
- Tamaño de población: 20
- Cantidad de generaciones: 5
- Probabilidad de cruce: 0.8
- Probabilidad de mutación: 0.3

GA Nivel 2:
- Tamaño de población: 50
- Cantidad de generaciones: 30
- Probabilidad de cruce: 0.8
- Probabilidad de mutación: 0.2

NSGA-II:
- Tamaño de población: 80
- Cantidad de generaciones: 40
- Objetivos: Maximizar proxy_ga, Minimizar params_k

Entrenamiento final:
- Fase 1: 5 arquitecturas x 50 épocas (evaluadas en Validación)
- Fase 2: 2 arquitecturas x 200 épocas (múltiples semillas, evaluadas en Test al final)
"""
    with open(metadata_file, 'w', encoding='utf-8') as f:
        f.write(metadata_initial)

def append_ga1_results(metadata_file, best_p50, best_weights_ga):
    with open(metadata_file, 'a', encoding='utf-8') as f:
        f.write(f"\n------------------------------------------------------------\n")
        f.write(f"RESULTADOS GA NIVEL 1\n")
        f.write(f"------------------------------------------------------------\n\n")
        f.write(f"Mejor fitness: {best_p50:.4f}\n")
        f.write(f"Pesos de los Zero-Cost Proxies:\n")
        for k, v in best_weights_ga.items():
            f.write(f"- {k}: {v:.4f}\n")

def append_ga2_results(metadata_file, mejor_fit_ga2, arch_str_ga2, acc_ga2, param_ga2):
    with open(metadata_file, 'a', encoding='utf-8') as f:
        f.write(f"\n------------------------------------------------------------\n")
        f.write(f"RESULTADOS GA NIVEL 2\n")
        f.write(f"------------------------------------------------------------\n\n")
        f.write(f"Mejor fitness: {mejor_fit_ga2:.4f}\n")
        f.write(f"Mejor arquitectura: {arch_str_ga2}\n")
        f.write(f"Accuracy del benchmark: {acc_ga2}\n")
        f.write(f"Cantidad de parámetros: {param_ga2}\n")

def append_nsga2_results(metadata_file, n_pareto, n_ev_nsga2, hv_nsga2=None, hv_real=None, hv_rs=None):
    with open(metadata_file, 'a', encoding='utf-8') as f:
        f.write(f"\n------------------------------------------------------------\n")
        f.write(f"RESULTADOS NSGA-II\n")
        f.write(f"------------------------------------------------------------\n\n")
        f.write(f"Cantidad de soluciones del frente de Pareto: {n_pareto}\n")
        f.write(f"Cantidad de evaluaciones: {n_ev_nsga2}\n")
        if hv_nsga2 is not None:
            f.write(f"Hipervolumen normalizado (HV NSGA-II): {hv_nsga2:.4f}\n")
        if hv_real is not None:
            f.write(f"Hipervolumen normalizado (HV Frente Real): {hv_real:.4f}\n")
        if hv_rs is not None:
            f.write(f"Hipervolumen normalizado (HV Random Search): {hv_rs:.4f}\n")

def append_phase1_results(metadata_file, phase1_results):
    with open(metadata_file, 'a', encoding='utf-8') as f:
        f.write(f"\n------------------------------------------------------------\n")
        f.write(f"FASE 1: 5 ARQUITECTURAS X 50 ÉPOCAS (VALIDACIÓN)\n")
        f.write(f"------------------------------------------------------------\n\n")
        for i, r in enumerate(phase1_results, 1):
            f.write(f"Arquitectura {i}:\n")
            f.write(f"- arch_str: {r['arch_str']}\n")
            f.write(f"- channels: {r['channel']}\n")
            f.write(f"- num_cells: {r['num_cells']}\n")
            f.write(f"- parámetros: {r['params_k']}\n")
            f.write(f"- mejor val_acc (50ep): {r['val_acc_50ep']:.2f}%\n")

def append_phase2_results(metadata_file, final_results, total_time):
    with open(metadata_file, 'a', encoding='utf-8') as f:
        f.write(f"\n------------------------------------------------------------\n")
        f.write(f"FASE 2: ENTRENAMIENTO FINAL (TEST EVALUATION)\n")
        f.write(f"------------------------------------------------------------\n\n")
        for i, r in enumerate(final_results, 1):
            f.write(f"Arquitectura final {i}:\n")
            f.write(f"- arch_str: {r['arch_str']}\n")
            f.write(f"- channels: {r['channel']}\n")
            f.write(f"- num_cells: {r['num_cells']}\n")
            f.write(f"- parámetros: {r['params_k']}\n")
            if 'seeds' in r and len(r['seeds']) > 1:
                f.write(f"- semillas evaluadas: {r['seeds']}\n")
                f.write(f"- test accuracy por semilla: {[round(x, 2) for x in r['test_accs']]}\n")
                f.write(f"- test accuracy media: {r['test_acc_mean']:.2f}% ± {r['test_acc_std']:.2f}%\n")
                f.write(f"- mejor val accuracy media: {r.get('val_acc_mean', 0.0):.2f}%\n")
            else:
                f.write(f"- test accuracy: {r.get('test_acc', r.get('acc_200ep', 0.0)):.2f}%\n")
                f.write(f"- mejor val accuracy: {r.get('best_val_acc', 0.0):.2f}%\n")
            f.write(f"- modelo guardado: {r.get('model_file', 'pt')}\n\n")

        f.write(f"Fecha y hora de finalización: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Tiempo total de ejecución: {total_time:.2f} segundos\n")
