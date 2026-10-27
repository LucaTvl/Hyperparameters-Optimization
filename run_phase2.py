import os
import sys
import json
import argparse
import time
import torch
import numpy as np

# Añadir raíz del proyecto al path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

# Soporte para codificación UTF-8 en consola de Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

from config.settings import DEVICE, OUTPUTS, CIFAR_DIR
from data.dataset import get_cifar10_loaders
from core.train import train_network, evaluate

CANDIDATES_FILE = os.path.join(OUTPUTS, "candidatas_fase2.json")
CHECKPOINTS_DIR = os.path.join(OUTPUTS, "checkpoints_fase2")
DEFAULT_SEEDS = [9, 67, 343]

def ensure_candidates_file():
    """Si no existe candidatas_fase2.json, crea uno con las mejores arquitecturas obtenidas."""
    if not os.path.exists(CANDIDATES_FILE):
        default_candidates = [
            {
                "id": 1,
                "arch_str": "|nor_conv_1x1~0|+|nor_conv_3x3~0|nor_conv_3x3~1|+|nor_conv_3x3~0|avg_pool_3x3~1|nor_conv_3x3~2|",
                "channel": 64,
                "num_cells": 9,
                "params_k": 1372.362,
                "proxy_score": 0.85
            },
            {
                "id": 2,
                "arch_str": "|nor_conv_3x3~0|+|none~0|skip_connect~1|+|none~0|nor_conv_3x3~1|nor_conv_1x1~2|",
                "channel": 64,
                "num_cells": 9,
                "params_k": 706.506,
                "proxy_score": 0.80
            }
        ]
        os.makedirs(OUTPUTS, exist_ok=True)
        with open(CANDIDATES_FILE, "w", encoding="utf-8") as f:
            json.dump(default_candidates, f, indent=2)
        print(f"[OK] Archivo de candidatas inicializado en: {CANDIDATES_FILE}")

def load_candidates():
    ensure_candidates_file()
    with open(CANDIDATES_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

def show_status(candidates, seeds):
    print("\n" + "=" * 65)
    print("ESTADO ACTUAL DE ENTRENAMIENTOS (FASE 2)")
    print("=" * 65)
    
    for cand in candidates:
        cid = cand.get("id", 1)
        print(f"\n[Arquitectura #{cid}] C={cand['channel']}, nc={cand['num_cells']}, params={cand['params_k']:.1f}K")
        print(f"Topología: {cand['arch_str']}")
        
        accs = []
        for s in seeds:
            final_path = os.path.join(CHECKPOINTS_DIR, f"arq_{cid}_seed_{s}_final.pt")
            in_prog_path = os.path.join(CHECKPOINTS_DIR, f"arq_{cid}_seed_{s}_in_progress.pt")
            
            if os.path.exists(final_path):
                ckpt = torch.load(final_path, map_location="cpu", weights_only=False)
                t_acc = ckpt.get("test_acc", ckpt.get("acc_200ep", 0.0))
                v_acc = ckpt.get("best_val_acc", 0.0)
                accs.append(t_acc)
                print(f"  - Semilla {s:4d}: [COMPLETA]  -> Test Acc: {t_acc:.2f}% | Mejor Val Acc: {v_acc:.2f}%")
            elif os.path.exists(in_prog_path):
                ckpt = torch.load(in_prog_path, map_location="cpu", weights_only=False)
                ep = ckpt.get("epoch", 0) + 1
                v_acc = ckpt.get("best_val_acc", 0.0)
                print(f"  - Semilla {s:4d}: [EN CURSO]   -> Época {ep}/200 | Val Acc actual: {v_acc:.2f}%")
            else:
                print(f"  - Semilla {s:4d}: [PENDIENTE]")
                
        if len(accs) > 0:
            mean = np.mean(accs)
            std = np.std(accs) if len(accs) > 1 else 0.0
            print(f"  >> Media Test ({len(accs)}/{len(seeds)} semillas): {mean:.2f}% ± {std:.2f}%")
    print("=" * 65 + "\n")

def run_seed(cand, seed, n_epochs, train_loader, val_loader, test_loader):
    cid = cand.get("id", 1)
    os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
    
    final_path = os.path.join(CHECKPOINTS_DIR, f"arq_{cid}_seed_{seed}_final.pt")
    in_prog_path = os.path.join(CHECKPOINTS_DIR, f"arq_{cid}_seed_{seed}_in_progress.pt")
    
    if os.path.exists(final_path):
        print(f"  -> Semilla {seed} ya está completada. Cargando resultados...")
        ckpt = torch.load(final_path, map_location="cpu", weights_only=False)
        return ckpt.get("test_acc", 0.0)
    
    print(f"\n" + "-" * 60)
    print(f"Entrenando Arq #{cid} (C={cand['channel']}, nc={cand['num_cells']}) con Semilla {seed}")
    print(f"Épocas: {n_epochs} | Device: {DEVICE}")
    print("-" * 60)
    
    # Fijar semillas para reproducibilidad
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        
    t0 = time.time()
    best_val_acc, hist, best_net = train_network(
        cand["arch_str"], cand["channel"], cand["num_cells"],
        train_loader, val_loader, n_epochs=n_epochs,
        ckpt_save_path=in_prog_path, resume=True
    )
    
    print("\n  -> Evaluando modelo en conjunto de TEST (una única vez al final)...")
    test_acc = evaluate(best_net, test_loader)
    elapsed = time.time() - t0
    
    print(f"  [OK] Semilla {seed} finalizada en {elapsed/60:.1f}m")
    print(f"    Mejor Val Acc: {best_val_acc:.2f}% | Test Acc final: {test_acc:.2f}%")
    
    # Guardar checkpoint final permanente
    torch.save({
        "arch_id": cid,
        "arch_str": cand["arch_str"],
        "channel": cand["channel"],
        "num_cells": cand["num_cells"],
        "params_k": cand["params_k"],
        "seed": seed,
        "n_epochs": n_epochs,
        "best_val_acc": best_val_acc,
        "test_acc": test_acc,
        "acc_200ep": test_acc,  # Clave compatible con demo_predict.py
        "history_val": hist,
        "history_200": hist,    # Clave compatible con gráficos existentes
        "model_state_dict": best_net.state_dict(),
        "elapsed_seconds": elapsed
    }, final_path)
    print(f"  [OK] Checkpoint permanente guardado en: {final_path}")
    
    # Limpiar archivo temporal in_progress
    if os.path.exists(in_prog_path):
        os.remove(in_prog_path)
        
    return test_acc

def main():
    parser = argparse.ArgumentParser(description="Ejecutar entrenamientos finales (Fase 2) de forma modular y con checkpoints.")
    parser.add_argument("--seed", type=int, default=None, help="Ejecutar únicamente una semilla específica (ej. 42, 123 o 999).")
    parser.add_argument("--all", action="store_true", help="Ejecutar todas las semillas pendientes secuencialmente.")
    parser.add_argument("--status", action="store_true", help="Mostrar el estado actual de los checkpoints sin entrenar.")
    parser.add_argument("--epochs", type=int, default=200, help="Cantidad de épocas de entrenamiento (default: 200).")
    parser.add_argument("--batch-size", type=int, default=96, help="Tamaño de batch para train (default: 96).")
    parser.add_argument("--arch-id", type=int, default=None, help="Filtrar por ID de arquitectura candidata (1 o 2).")
    args = parser.parse_args()

    candidates = load_candidates()
    seeds = DEFAULT_SEEDS

    if args.status or (args.seed is None and not args.all):
        show_status(candidates, seeds)
        if not args.status and args.seed is None and not args.all:
            print("Para entrenar podés usar:")
            print("  python run_phase2.py --all           (ejecuta todas las semillas pendientes)")
            print("  python run_phase2.py --seed 42       (ejecuta solo la semilla 42)")
            print("  python run_phase2.py --seed 123      (ejecuta solo la semilla 123)")
            print("  python run_phase2.py --status        (solo consulta el avance)")
        return

    # Si se especificó una arquitectura particular
    if args.arch_id is not None:
        candidates = [c for c in candidates if c.get("id") == args.arch_id]
        if not candidates:
            print(f"Error: No se encontró la arquitectura con id={args.arch_id}")
            return

    seeds_to_run = [args.seed] if args.seed is not None else seeds

    print(f"\nCargando DataLoaders (Train 45k, Val 5k, Test 10k)...")
    train_loader, val_loader, test_loader = get_cifar10_loaders(train_batch=args.batch_size)
    print("✓ Datos listos.")

    for cand in candidates:
        for s in seeds_to_run:
            run_seed(cand, s, args.epochs, train_loader, val_loader, test_loader)

    # Mostrar resumen final actualizado
    show_status(load_candidates(), seeds)

if __name__ == "__main__":
    main()
