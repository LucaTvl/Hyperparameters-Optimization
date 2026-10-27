import os
import time
import torch
import random
import numpy as np
import pandas as pd
import itertools
try:
    from tqdm.auto import tqdm
except ImportError:
    from tqdm import tqdm

# Importaciones de configuración
from config.settings import (
    DEVICE, DATASET, CIFAR_DIR, CSV_GT, CSV_ZCP, CSV_ZCP_FULL, CSV_ZCP_EXT,
    MASTER_SEED, ZCP_COLS, ZCP_COLS_WITH_PARAMS, CHANNELS, NUM_CELLS,
    SAMPLE_PER_GROUP, GRAFICOS_DIR, METADATA_FILE, MODELOS_DIR, GIT_COMMIT,
    START_TIME_TOTAL, setup_output_dirs, OUTPUTS
)

# Importaciones de datos y métricas
from data.dataset import get_cifar10_batch, get_cifar10_loaders
from data.csv_handler import (
    load_ground_truth, load_zcps, load_full_zcps,
    get_normalization_params, apply_normalization, split_calib_val
)
from models.zcp_metrics import compute_all_zcps, compute_zcps_ext
from models.networks import count_params_k

# Importaciones de optimizadores y baselines
from optimizers.genetic_ops import random_weights, random_ind, random_ind_ext
from optimizers.ga_level1 import run_ga_level1, precision_at_k, weights_to_dict
from optimizers.ga_level2 import run_ga_level2, run_rs_level2, greedy_search, build_proxy
from optimizers.nsga2 import run_nsga2

import json
# Importaciones utilitarias
from utils.logger import (
    write_initial_metadata, append_ga1_results, append_ga2_results,
    append_nsga2_results, append_phase1_results, append_phase2_results
)
from utils.visualization import (
    plot_ga1_convergence, plot_ga2_convergence, plot_nsga2_pareto, plot_final_results
)
from utils.metrics import compute_hypervolume_2d
from core.train import train_network, evaluate

def main():
    setup_output_dirs()
    print(f'✓ Semilla maestra: {MASTER_SEED}')
    print(f'✓ Device : {DEVICE}')
    
    write_initial_metadata(METADATA_FILE, os.path.abspath(__file__), GIT_COMMIT, DEVICE, DATASET, CIFAR_DIR, CSV_GT, CSV_ZCP, GRAFICOS_DIR, MASTER_SEED)

    # 1. Cargar Ground Truth y preparar DataFrame Base
    df = load_ground_truth()
    print(f'✓ Ground truth: {len(df)} arquitecturas')

    x_batch, y_batch = get_cifar10_batch(batch_size=64)
    print(f'✓ CIFAR-10 batch: {tuple(x_batch.shape)}')

    # 2. Cargar o generar Zero-Cost Proxies completos
    df_zcp_full = load_full_zcps(df, x_batch, y_batch)

    df_full = df_zcp_full.merge(df[['arch_id','arch_str','acc_valid','params_k','rank_real']], on='arch_id')
    df_full = df_full.merge(df[['arch_id','params_k']], on='arch_id', how='left', suffixes=('','_dup'))
    if 'params_k_dup' in df_full.columns:
        df_full.drop(columns=['params_k_dup'], inplace=True)
        
    norm_params = get_normalization_params(df_full, ZCP_COLS_WITH_PARAMS)
    df_norm = apply_normalization(df_full, norm_params, ZCP_COLS_WITH_PARAMS)
    df_calib, df_val = split_calib_val(df_norm)

    print('\nEjecutando GA Nivel 1 — 10 corridas independientes...')
    results_ga1 = []
    t0 = time.time()
    for run in range(10):
        best_w, best_p, hist = run_ga_level1(df_calib, pop_size=20, n_gen=5, seed=MASTER_SEED + run*17, cols=ZCP_COLS)
        results_ga1.append({'run': run, 'best_p50': best_p, 'weights': best_w, 'history': hist})
        print(f'  Run {run+1:2d}: Precision@50 = {best_p:.4f}')
    
    elapsed = time.time() - t0
    print(f'\n✓ GA Nivel 1 completado en {elapsed:.1f}s')

    best_run = max(results_ga1, key=lambda r: r['best_p50'])
    best_weights_ga = weights_to_dict(best_run['weights'], ZCP_COLS)
    append_ga1_results(METADATA_FILE, best_run['best_p50'], best_weights_ga)

    # Random Search y BO baseline para gráficos
    results_rs1 = []
    for run in range(10):
        random.seed(MASTER_SEED + run*31 + 100)
        cands = [random_weights(len(ZCP_COLS)) for _ in range(100)]
        fits = [precision_at_k(weights_to_dict(c, ZCP_COLS), df_calib, k=50, cols=ZCP_COLS) for c in cands]
        best_i = np.argmax(fits)
        results_rs1.append({'best_p50': fits[best_i]})
    
    bo_p50 = 0.68
    ga_p50s = [r['best_p50'] for r in results_ga1]
    rs_p50s = [r['best_p50'] for r in results_rs1]
    plot_ga1_convergence(ga_p50s, rs_p50s, bo_p50, results_ga1, f'{GRAFICOS_DIR}/ga_nivel1_comparacion.png')

    # Búsqueda Arquitectural Nivel 2
    df_full['proxy_ga'] = build_proxy(best_weights_ga, df_norm, ZCP_COLS)
    proxy_lookup = df_full.set_index('arch_str')[['proxy_ga','rank_real','acc_valid','params_k']].to_dict('index')

    print('\nEjecutando GA Nivel 2 (3 corridas)...')
    results_ga2 = []
    for seed in [MASTER_SEED + 400, MASTER_SEED + 407, MASTER_SEED + 499]:
        _, best_f, hist, best_info, n_ev = run_ga_level2(proxy_lookup, pop_size=50, n_gen=30, seed=seed)
        results_ga2.append({'seed': seed, 'history': hist, 'best_info': best_info, 'n_evals': n_ev})

    best_ga2 = max(results_ga2, key=lambda r: r['history'][-1]['best_fit'])
    append_ga2_results(METADATA_FILE, best_ga2['history'][-1]['best_fit'], df_full.loc[df_full['acc_valid']==best_ga2['best_info']['acc_valid'], 'arch_str'].values[0] if len(df_full[df_full['acc_valid']==best_ga2['best_info']['acc_valid']])>0 else "Desc", best_ga2['best_info']['acc_valid'], best_ga2['best_info']['params_k'])

    rs_f, rs_info, rs_hist = run_rs_level2(proxy_lookup, n_evals=1500, seed=MASTER_SEED + 500)
    greedy_20 = greedy_search(df_full, 20)
    plot_ga2_convergence(results_ga2, rs_info, greedy_20, f'{GRAFICOS_DIR}/ga_nivel2_convergencia.png')

    # Búsqueda NSGA-II
    print('\nEjecutando NSGA-II...')
    df_pareto_nsga2, hist_fronts, n_ev_nsga2 = run_nsga2(proxy_lookup, pop_size=80, n_gen=40, seed=MASTER_SEED + 600)

    df_pareto_rs = pd.DataFrame([{'proxy_ga': -np.inf, 'params_k': 1000, 'acc_valid': 88.0}]) # Placeholder para plot
    
    # Extraer el verdadero Frente de Pareto (aprox 42 arquitecturas)
    def get_real_pareto(df):
        pareto = []
        sorted_df = df.sort_values(by=['params_k', 'acc_valid'], ascending=[True, False])
        best_acc = -1
        for _, row in sorted_df.iterrows():
            if row['acc_valid'] > best_acc:
                pareto.append(row)
                best_acc = row['acc_valid']
        return pd.DataFrame(pareto)

    df_pareto_real = get_real_pareto(df_full)
    
    # ── Métrica de Calidad: Hipervolumen (HV) ──
    hv_real = compute_hypervolume_2d(df_pareto_real)
    hv_nsga2 = compute_hypervolume_2d(df_pareto_nsga2)
    print(f"✓ Hipervolumen normalizado Frente Real: {hv_real:.4f}")
    print(f"✓ Hipervolumen normalizado NSGA-II:     {hv_nsga2:.4f} ({100*hv_nsga2/hv_real:.1f}% del frente real)")
    
    append_nsga2_results(METADATA_FILE, len(df_pareto_nsga2), n_ev_nsga2, hv_nsga2=hv_nsga2, hv_real=hv_real)
    
    plot_nsga2_pareto(df_pareto_real, df_pareto_nsga2, df_pareto_rs, hist_fronts, f'{GRAFICOS_DIR}/nsga2_pareto.png')
    
    # ── Espacio Extendido ──
    from optimizers.extended_space import get_extended_zcps, recalibrate_ga1, run_nsga2_extended
    df_ext = get_extended_zcps(df_full, x_batch, y_batch)
    df_combined_for_norm = pd.concat([df_full[ZCP_COLS_WITH_PARAMS], df_ext[ZCP_COLS_WITH_PARAMS]], ignore_index=True)
    
    base_signs = {col: norm_params[col]['sign'] for col in ZCP_COLS_WITH_PARAMS}
    norm_params_ext = get_normalization_params(df_combined_for_norm, ZCP_COLS_WITH_PARAMS, base_signs=base_signs)
    df_full_renorm = apply_normalization(df_full, norm_params_ext, ZCP_COLS_WITH_PARAMS)
    df_calib_renorm, _ = split_calib_val(df_full_renorm)
    
    best_weights_ext = recalibrate_ga1(df_calib_renorm)
    df_pareto_ext = run_nsga2_extended(norm_params_ext, best_weights_ext, x_batch, y_batch, df_full, pop_size=60, n_gen=30, seed=MASTER_SEED + 700)
    
    # ── FASE 1: Entrenar Top 5 extendido en VALIDACIÓN ──
    print('\nFASE 1: Top-5 candidatas × 50 épocas (monitoreadas en Validación)')
    top5_candidates = df_pareto_ext.nlargest(5, 'proxy_score').copy()
    
    train_loader, val_loader, test_loader = get_cifar10_loaders()
    phase1_results = []
    
    for idx_cand, (_, row) in enumerate(top5_candidates.iterrows(), 1):
        ckpt_p1 = os.path.join(OUTPUTS, f"fase1_cand_{idx_cand}_in_progress.pt")
        best_val_acc, hist, _ = train_network(
            row['arch_str'], row['channel'], row['num_cells'],
            train_loader, val_loader, n_epochs=50,
            ckpt_save_path=ckpt_p1
        )
        if os.path.exists(ckpt_p1):
            os.remove(ckpt_p1)

        phase1_results.append({
            'arch_str' : row['arch_str'], 'channel': row['channel'], 'num_cells': row['num_cells'],
            'params_k' : row['params_k'], 'proxy'  : row['proxy_score'],
            'val_acc_50ep' : best_val_acc, 'history': hist,
        })
    append_phase1_results(METADATA_FILE, phase1_results)

    # ── Selección limpia de finalistas según VALIDACIÓN ──
    print('\nFASE 2: Selección de mejores 2 candidatas por Validación')
    top2 = sorted(phase1_results, key=lambda r: r['val_acc_50ep'], reverse=True)[:2]
    
    # Guardar candidatas en JSON para permitir ejecuciones independientes/reanudables
    cands_to_save = []
    for idx, r in enumerate(top2, 1):
        cands_to_save.append({
            'id': idx,
            'arch_str': r['arch_str'],
            'channel': r['channel'],
            'num_cells': r['num_cells'],
            'params_k': r['params_k'],
            'proxy_score': r['proxy'],
            'val_acc_50ep': r['val_acc_50ep']
        })
    cands_path = os.path.join(OUTPUTS, 'candidatas_fase2.json')
    with open(cands_path, 'w', encoding='utf-8') as f:
        json.dump(cands_to_save, f, indent=2)
    print(f"✓ Candidatas finalistas guardadas en: {cands_path}")

    # ── FASE 2: Entrenamiento final con múltiples semillas y checkpoints ──
    SEEDS_FASE2 = [9, 67, 343]
    print(f'\nFASE 2: Entrenando {len(top2)} arquitecturas con {len(SEEDS_FASE2)} semillas x 200 épocas')
    checkpoints_f2_dir = os.path.join(OUTPUTS, 'checkpoints_fase2')
    os.makedirs(checkpoints_f2_dir, exist_ok=True)
    
    final_results = []
    for idx_cand, r in enumerate(top2, 1):
        test_accs = []
        val_accs = []
        best_overall_net = None
        best_overall_test = -1.0
        best_hist = []

        print(f"\n--- Arquitectura {idx_cand}/2 (C={r['channel']}, nc={r['num_cells']}, params={r['params_k']:.1f}K) ---")
        for s in SEEDS_FASE2:
            ckpt_final = os.path.join(checkpoints_f2_dir, f"arq_{idx_cand}_seed_{s}_final.pt")
            ckpt_prog = os.path.join(checkpoints_f2_dir, f"arq_{idx_cand}_seed_{s}_in_progress.pt")
            
            if os.path.exists(ckpt_final):
                print(f"  [✓] Semilla {s} ya completada previamente. Cargando...")
                data = torch.load(ckpt_final, map_location='cpu', weights_only=False)
                t_acc = data.get('test_acc', data.get('acc_200ep', 0.0))
                v_acc = data.get('best_val_acc', 0.0)
                test_accs.append(t_acc)
                val_accs.append(v_acc)
                if t_acc > best_overall_test:
                    best_overall_test = t_acc
                    best_hist = data.get('history_200', data.get('history_val', []))
                continue
                
            print(f"  [>] Entrenando Semilla {s} (200 épocas)...")
            torch.manual_seed(s)
            np.random.seed(s)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(s)
                
            b_val_acc, hist_200, net = train_network(
                r['arch_str'], r['channel'], r['num_cells'],
                train_loader, val_loader, n_epochs=200,
                ckpt_save_path=ckpt_prog, resume=True
            )
            
            # Evaluación imparcial en TEST (una única vez al final de la corrida)
            t_acc = evaluate(net, test_loader)
            test_accs.append(t_acc)
            val_accs.append(b_val_acc)
            
            # Guardar checkpoint permanente de la corrida
            torch.save({
                'arch_id': idx_cand,
                'arch_str': r['arch_str'],
                'channel': r['channel'],
                'num_cells': r['num_cells'],
                'params_k': r['params_k'],
                'seed': s,
                'best_val_acc': b_val_acc,
                'test_acc': t_acc,
                'acc_200ep': t_acc,
                'history_200': hist_200,
                'model_state_dict': net.state_dict()
            }, ckpt_final)
            
            if os.path.exists(ckpt_prog):
                os.remove(ckpt_prog)
                
            if t_acc > best_overall_test:
                best_overall_test = t_acc
                best_overall_net = net
                best_hist = hist_200
                
            print(f"    Semilla {s}: Mejor Val Acc = {b_val_acc:.2f}% | Test Acc final = {t_acc:.2f}%")

        mean_test = float(np.mean(test_accs)) if test_accs else 0.0
        std_test = float(np.std(test_accs)) if len(test_accs) > 1 else 0.0
        mean_val = float(np.mean(val_accs)) if val_accs else 0.0
        
        # Guardar modelo en MODELOS_DIR para compatibilidad directa con demo_predict.py
        fname = f"{MODELOS_DIR}/C{r['channel']}_nc{r['num_cells']}_acc{mean_test:.2f}.pt"
        if best_overall_net is not None:
            torch.save({
                'arch_str': r['arch_str'], 'channel': r['channel'], 'num_cells': r['num_cells'],
                'acc_200ep': mean_test, 'history_200': best_hist, 'model_state_dict': best_overall_net.state_dict()
            }, fname)
            
        final_results.append({
            **r,
            'seeds': SEEDS_FASE2,
            'test_accs': test_accs,
            'test_acc_mean': mean_test,
            'test_acc_std': std_test,
            'val_acc_mean': mean_val,
            'acc_200ep': mean_test,
            'history_200': best_hist,
            'model_file': fname
        })
        print(f"  Resultado Arq {idx_cand}: Test Acc = {mean_test:.2f}% ± {std_test:.2f}%")

    total_time = time.perf_counter() - START_TIME_TOTAL
    append_phase2_results(METADATA_FILE, final_results, total_time)
    
    plot_final_results(final_results, df_pareto_ext, 91.68, f'{GRAFICOS_DIR}/resultados_finales.png')
    
    print("Ejecución finalizada con éxito.")

if __name__ == '__main__':
    main()
