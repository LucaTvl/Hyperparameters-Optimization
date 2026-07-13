import torch
import torch.nn as nn
from config.settings import DEVICE

class ReLUConvBN(nn.Module):
    def __init__(self, C, kernel, stride, pad):
        super().__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C, C, kernel, stride=stride, padding=pad, bias=False),
            nn.BatchNorm2d(C, affine=True),
        )
    def forward(self, x): return self.op(x)

class Zero(nn.Module):
    def __init__(self, stride):
        super().__init__()
        self.stride = stride
    def forward(self, x):
        if self.stride == 1: return x.mul(0.)
        return x[:, :, ::self.stride, ::self.stride].mul(0.)

class FactorizedReduce(nn.Module):
    def __init__(self, C):
        super().__init__()
        self.relu  = nn.ReLU(inplace=False)
        self.conv1 = nn.Conv2d(C, C//2, 1, stride=2, padding=0, bias=False)
        self.conv2 = nn.Conv2d(C, C//2, 1, stride=2, padding=0, bias=False)
        self.bn    = nn.BatchNorm2d(C, affine=True)
    def forward(self, x):
        x = self.relu(x)
        return self.bn(torch.cat([self.conv1(x), self.conv2(x[:,:,1:,1:])], dim=1))

def make_op(op_name, C, stride):
    if op_name == 'none':         return Zero(stride)
    if op_name == 'skip_connect': return nn.Identity() if stride==1 else FactorizedReduce(C)
    if op_name == 'nor_conv_1x1': return ReLUConvBN(C, 1, stride, 0)
    if op_name == 'nor_conv_3x3': return ReLUConvBN(C, 3, stride, 1)
    if op_name == 'avg_pool_3x3': return nn.AvgPool2d(3, stride=stride, padding=1, count_include_pad=False)
    raise ValueError(f'Op desconocida: {op_name}')

class NATSCell(nn.Module):
    def __init__(self, arch_str, C, stride):
        super().__init__()
        self.ops, self.edges = nn.ModuleList(), []
        for dst, block in enumerate(arch_str.split('+'), start=1):
            for part in [p for p in block.split('|') if p.strip()]:
                op_name, src = part.strip().split('~')
                src = int(src)
                self.ops.append(make_op(op_name, C, stride if src==0 else 1))
                self.edges.append((src, dst))
    def forward(self, x):
        nodes = [x]
        for dst in range(1, 4):
            inputs = [self.ops[i](nodes[src]) for i, (src, d) in enumerate(self.edges) if d == dst]
            nodes.append(sum(inputs))
        return nodes[-1]

class TinyNetwork(nn.Module):
    def __init__(self, arch_str, n_classes=10, C=16):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(3,C,3,padding=1,bias=False), nn.BatchNorm2d(C))
        self.cells = nn.ModuleList([NATSCell(arch_str,C,s) for s in [1,2,1,2,1]])
        self.lastact = nn.Sequential(nn.BatchNorm2d(C), nn.ReLU(inplace=False))
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(C, n_classes)
    def forward(self, x):
        x = self.stem(x)
        for cell in self.cells: x = cell(x)
        x = self.lastact(x)
        x = self.gap(x).view(x.size(0), -1)
        return self.classifier(x), None

def build_network(arch_str, n_classes=10, C=16):
    return TinyNetwork(arch_str, n_classes=n_classes, C=C)

def build_network_ext(arch_str, n_classes=10, C=16, num_cells=5):
    """
    Construye la red para el espacio de búsqueda extendido, 
    permitiendo variar canales y número de celdas.
    """
    class TinyNetExt(nn.Module):
        def __init__(self):
            super().__init__()
            self.stem = nn.Sequential(
                nn.Conv2d(3, C, 3, padding=1, bias=False),
                nn.BatchNorm2d(C)
            )
            strides = []
            for i in range(num_cells):
                if num_cells <= 3:
                    strides.append(2 if i == 1 else 1)
                else:
                    strides.append(2 if i in [1, num_cells//2+1] else 1)
            self.cells = nn.ModuleList([NATSCell(arch_str, C, s) for s in strides])
            self.lastact = nn.Sequential(nn.BatchNorm2d(C), nn.ReLU(inplace=False))
            self.gap = nn.AdaptiveAvgPool2d(1)
            self.classifier = nn.Linear(C, n_classes)
        def forward(self, x):
            x = self.stem(x)
            for cell in self.cells: x = cell(x)
            x = self.lastact(x)
            x = self.gap(x).view(x.size(0), -1)
            return self.classifier(x), None
            
    return TinyNetExt().to(DEVICE)

def count_params_k(arch_str, C, num_cells):
    """Cuenta el número de parámetros de la red en miles (K)."""
    net = build_network_ext(arch_str, C=C, num_cells=num_cells)
    return sum(p.numel() for p in net.parameters()) / 1000
