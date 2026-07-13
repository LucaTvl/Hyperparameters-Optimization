import torch
import torch.nn as nn
import numpy as np
from config.settings import DEVICE
from models.networks import build_network, build_network_ext

criterion = nn.CrossEntropyLoss()

def zcp_grad_norm(net, x, y):
    net.zero_grad()
    criterion(net(x)[0], y).backward()
    return sum(p.grad.norm().item()**2 for p in net.parameters() if p.grad is not None)**0.5

def zcp_snip(net, x, y):
    net.zero_grad()
    criterion(net(x)[0], y).backward()
    return sum((p.grad*p.data).abs().sum().item() for p in net.parameters() if p.grad is not None)

def zcp_grasp(net, x, y, T=1):
    net.zero_grad()
    criterion(net(x)[0]/T, y).backward(retain_graph=True)
    g1 = {n: p.grad.clone() for n,p in net.named_parameters() if p.grad is not None}
    net.zero_grad()
    criterion(net(x)[0]/T, y).backward()
    return sum(-(g1[n]*p.grad*p.data).sum().item() for n,p in net.named_parameters() if p.grad is not None and n in g1)

def zcp_jacob_cov(net, x):
    net.zero_grad()
    x_in = x.clone().detach().requires_grad_(True)
    net(x_in)[0].backward(torch.ones(x.shape[0], 10).to(x.device))
    J = x_in.grad.detach().reshape(x.shape[0], -1).cpu().numpy()
    eigv = np.abs(np.linalg.eigvalsh(J @ J.T))
    return float(np.log(eigv + 1e-5).sum())

def zcp_synflow(net, x):
    ones = torch.ones_like(x)
    signs = [p.data.sign().clone() for p in net.parameters()]
    for p in net.parameters(): p.data.abs_()
    net.zero_grad()
    net(ones)[0].sum().backward()
    score = sum((p.data*p.grad).sum().item() for p in net.parameters() if p.grad is not None)
    for p, s in zip(net.parameters(), signs): p.data.mul_(s)
    return score

def compute_all_zcps(arch_str, x, y, C=16):
    """
    Computa el conjunto de Zero-Cost Proxies para una arquitectura en el espacio base.
    """
    try:
        net = build_network(arch_str, C=C).to(DEVICE).train()
        torch.manual_seed(hash(arch_str) % (2**31))
        for p in net.parameters():
            if p.dim() > 1: nn.init.kaiming_normal_(p)
        return {
            'grad_norm' : zcp_grad_norm(net, x, y),
            'snip'      : zcp_snip(net, x, y),
            'grasp'     : zcp_grasp(net, x, y),
            'jacob_cov' : zcp_jacob_cov(net, x),
            'synflow'   : zcp_synflow(net, x),
        }
    except Exception:
        return None

def compute_zcps_ext(arch_str, C, num_cells, x, y):
    """
    Computa el conjunto de Zero-Cost Proxies para una arquitectura en el espacio extendido.
    """
    try:
        net = build_network_ext(arch_str, C=C, num_cells=num_cells).train()
        torch.manual_seed(hash(arch_str + str(C) + str(num_cells)) % (2**31))
        for p in net.parameters():
            if p.dim() > 1: nn.init.kaiming_normal_(p)
        return {
            'grad_norm' : zcp_grad_norm(net, x, y),
            'snip'      : zcp_snip(net, x, y),
            'grasp'     : zcp_grasp(net, x, y),
            'jacob_cov' : zcp_jacob_cov(net, x),
            'synflow'   : zcp_synflow(net, x),
        }
    except Exception:
        return None
