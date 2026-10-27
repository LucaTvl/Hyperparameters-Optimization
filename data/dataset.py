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

def get_cifar10_loaders(train_batch=96, eval_batch=256, val_size=5000, seed=42):
    """
    Retorna los DataLoaders separados de entrenamiento, validación y test.
    - Train (45.000 imgs): train=True con data augmentation.
    - Val (5.000 imgs): subconjunto de train=True SIN data augmentation.
    - Test (10.000 imgs): train=False oficial, para evaluación única final.
    """
    transform_train = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    
    transform_eval = T.Compose([
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])

    # Conjunto de train con y sin aumentación para separar train y validación
    full_train_augmented = torchvision.datasets.CIFAR10(
        CIFAR_DIR, train=True, download=True, transform=transform_train
    )
    full_train_clean = torchvision.datasets.CIFAR10(
        CIFAR_DIR, train=True, download=False, transform=transform_eval
    )

    # Partición determinista por semilla fija
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(full_train_augmented), generator=generator).tolist()
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]

    train_set = torch.utils.data.Subset(full_train_augmented, train_indices)
    val_set = torch.utils.data.Subset(full_train_clean, val_indices)

    # Test oficial aislado (10.000 muestras)
    test_set = torchvision.datasets.CIFAR10(
        CIFAR_DIR, train=False, download=True, transform=transform_eval
    )

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=train_batch, shuffle=True,
        num_workers=0, pin_memory=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=eval_batch, shuffle=False,
        num_workers=0, pin_memory=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=eval_batch, shuffle=False,
        num_workers=0, pin_memory=True
    )
    
    return train_loader, val_loader, test_loader
