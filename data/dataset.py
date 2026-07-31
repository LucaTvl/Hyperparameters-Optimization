import torch
import torchvision
import torchvision.transforms as T
import warnings
import numpy as np

from config.settings import CIFAR_DIR, DEVICE

# Suprimir warning deprecado de numpy al cargar CIFAR en torchvision
warnings.filterwarnings("ignore", category=np.exceptions.VisibleDeprecationWarning)

def get_cifar10_batch(batch_size=64):
    """
    Obtiene un único mini-batch de CIFAR-10.
    Útil para el cómputo inicial y rápido de los Zero-Cost Proxies.
    """
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])
    
    cifar_train = torchvision.datasets.CIFAR10(
        CIFAR_DIR, train=True, download=True, transform=transform
    )
    loader = torch.utils.data.DataLoader(
        cifar_train, batch_size=batch_size, shuffle=True, num_workers=0
    )
    
    x_batch, y_batch = next(iter(loader))
    x_batch = x_batch.to(DEVICE)
    y_batch = y_batch.to(DEVICE)
    
    return x_batch, y_batch

def get_cifar10_loaders(train_batch=96, val_batch=256):
    """
    Retorna los DataLoaders completos (entrenamiento y validación) 
    para CIFAR-10 con augmentación de datos, listos para entrenamiento real.
    """
    transform_train = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    
    transform_test = T.Compose([
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])

    train_set = torchvision.datasets.CIFAR10(
        CIFAR_DIR, train=True, download=True, transform=transform_train
    )
    val_set = torchvision.datasets.CIFAR10(
        CIFAR_DIR, train=False, download=True, transform=transform_test
    )

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=train_batch, shuffle=True,
        num_workers=0, pin_memory=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=val_batch, shuffle=False,
        num_workers=0, pin_memory=True
    )
    
    return train_loader, val_loader
