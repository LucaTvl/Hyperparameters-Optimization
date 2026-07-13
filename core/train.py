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

def train_network(arch_str, C, num_cells, train_loader, val_loader, n_epochs=50, lr=0.025):
    """
    Entrena una TinyNetExt por un número especificado de épocas y 
    retorna el mayor acc_valid, el historial de accuracies y la red entrenada.
    """
    net = build_network_ext(arch_str, C=C, num_cells=num_cells)
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=3e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=n_epochs)
    criterion = nn.CrossEntropyLoss()

    history = []
    
    pbar = tqdm(range(n_epochs), desc=f"Train (C={C}, nc={num_cells})", unit="ep")
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
        net.eval()
        correct = total = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                preds = net(xb)[0].argmax(1)
                correct += (preds == yb).sum().item()
                total += yb.size(0)
        acc = 100 * correct / total
        history.append(acc)
        
        pbar.set_postfix(acc=f"{acc:.2f}%")

        if (epoch+1) % 10 == 0:
            tqdm.write(f'    ep {epoch+1:3d}/{n_epochs}: acc={acc:.2f}%')

    return max(history), history, net
