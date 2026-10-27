import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
try:
    from tqdm.auto import tqdm
except ImportError:
    from tqdm import tqdm
from config.settings import DEVICE
from models.networks import build_network_ext

def evaluate(net, loader):
    """
    Evalúa una red en un DataLoader (validación o test) sin calcular gradientes.
    Retorna la precisión (accuracy) en porcentaje (0-100%).
    """
    net.eval()
    correct = total = 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            preds = net(xb)[0].argmax(1)
            correct += (preds == yb).sum().item()
            total += yb.size(0)
    return 100.0 * correct / total if total > 0 else 0.0

def train_network(arch_str, C, num_cells, train_loader, val_loader, n_epochs=50, lr=0.025, ckpt_save_path=None, resume=True):
    """
    Entrena una TinyNetExt monitoreando el conjunto de validación.
    - Guarda checkpoint periódico para permitir reanudar ante interrupciones.
    - Restaura los pesos correspondientes a la mejor época de validación antes de retornar.
    Retorna: (best_val_acc, history, best_net)
    """
    net = build_network_ext(arch_str, C=C, num_cells=num_cells)
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=3e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=n_epochs)
    criterion = nn.CrossEntropyLoss()

    start_epoch = 0
    history = []
    best_val_acc = -1.0
    best_state_dict = None

    # Reanudar desde checkpoint previo si existe y resume=True
    if resume and ckpt_save_path and os.path.exists(ckpt_save_path):
        try:
            print(f"  -> Reanudando entrenamiento desde checkpoint: {ckpt_save_path}")
            ckpt = torch.load(ckpt_save_path, map_location=DEVICE, weights_only=False)
            net.load_state_dict(ckpt['net'])
            optimizer.load_state_dict(ckpt['optimizer'])
            scheduler.load_state_dict(ckpt['scheduler'])
            start_epoch = ckpt['epoch'] + 1
            history = ckpt.get('history', [])
            best_val_acc = ckpt.get('best_val_acc', -1.0)
            best_state_dict = ckpt.get('best_state_dict', None)
            print(f"     Reanudado en época {start_epoch+1}/{n_epochs} (Mejor Val Acc previa: {best_val_acc:.2f}%)")
        except Exception as e:
            print(f"  [Aviso] No se pudo reanudar el checkpoint ({e}), comenzando desde época 1.")

    pbar = tqdm(range(start_epoch, n_epochs), desc=f"Train (C={C}, nc={num_cells})", unit="ep")
    for epoch in pbar:
        # ── Fase de Entrenamiento ──
        net.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            out, _ = net(xb)
            criterion(out, yb).backward()
            optimizer.step()
        scheduler.step()

        # ── Fase de Validación ──
        acc_val = evaluate(net, val_loader)
        history.append(acc_val)

        if acc_val > best_val_acc:
            best_val_acc = acc_val
            best_state_dict = {k: v.cpu().clone() for k, v in net.state_dict().items()}

        pbar.set_postfix(val_acc=f"{acc_val:.2f}%", best_val=f"{best_val_acc:.2f}%")

        if (epoch + 1) % 10 == 0 or epoch == n_epochs - 1:
            if (epoch + 1) % 10 == 0:
                tqdm.write(f'    ep {epoch+1:3d}/{n_epochs}: val_acc={acc_val:.2f}% (mejor={best_val_acc:.2f}%)')
            # Guardar checkpoint intermedio si se especificó ruta
            if ckpt_save_path:
                torch.save({
                    'epoch': epoch,
                    'net': net.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'history': history,
                    'best_val_acc': best_val_acc,
                    'best_state_dict': best_state_dict
                }, ckpt_save_path)

    # Cargar el modelo con la mejor validación encontrada
    if best_state_dict is not None:
        net.load_state_dict(best_state_dict)

    return best_val_acc, history, net
