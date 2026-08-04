import argparse
import os
import sys
import random
import warnings
import time
from datetime import datetime
import torch

# Suprimir el warning de deprecación específico de numpy/torchvision al cargar CIFAR-10
warnings.filterwarnings("ignore", message=".*align should be passed as Python or NumPy boolean.*")
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset

from models.networks import build_network_ext
from config.settings import DEVICE, CIFAR_DIR

CLASSES = [
    "avión", "auto", "pájaro", "gato", "ciervo",
    "perro", "rana", "caballo", "barco", "camión"
]

def main():
    parser = argparse.ArgumentParser(description="Probar o evaluar un modelo .pt en CIFAR-10")
    parser.add_argument("--model", type=str, required=True, help="Ruta al modelo .pt")
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--idx", type=int, help="Índice de la imagen en el test set")
    group.add_argument("--random", action="store_true", help="Elegir una imagen aleatoria")
    group.add_argument("--image", type=str, help="Ruta a una imagen local externa (.jpg, .png, etc.)")
    group.add_argument("--eval-n", type=int, help="Evaluar N imágenes del test set")
    group.add_argument("--eval-all", action="store_true", help="Evaluar todo el test set")
    
    parser.add_argument("--eval-seed", type=int, default=123, help="Seed para seleccionar muestra aleatoria en eval-n (default: 123)")
    parser.add_argument("--batch-size", type=int, default=256, help="Tamaño de batch para evaluación masiva (default: 256)")
    
    args = parser.parse_args()

    if not os.path.exists(args.model):
        print(f"Error: El archivo de modelo '{args.model}' no existe.")
        sys.exit(1)

    print(f"Ruta del modelo cargado: {args.model}")
    checkpoint = torch.load(args.model, map_location=DEVICE, weights_only=False)

    required_keys = ["arch_str", "channel", "num_cells", "model_state_dict"]
    for key in required_keys:
        if key not in checkpoint:
            print(f"Error: Falta la clave '{key}' dentro del checkpoint.")
            sys.exit(1)

    print(f"Canales: {checkpoint['channel']}")
    print(f"Cantidad de celdas: {checkpoint['num_cells']}")
    acc_saved = checkpoint.get("acc_200ep", None)
    if acc_saved is not None:
        print(f"Accuracy guardada del modelo: {acc_saved}")
    else:
        print("Accuracy guardada del modelo: No disponible")

    net = build_network_ext(
        checkpoint["arch_str"],
        C=checkpoint["channel"],
        num_cells=checkpoint["num_cells"]
    )
    
    net.load_state_dict(checkpoint["model_state_dict"])
    net.to(DEVICE)
    net.eval()

    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2470, 0.2435, 0.2616)

    # Modo predicción con imagen externa
    if args.image is not None:
        if not os.path.exists(args.image):
            print(f"Error: El archivo de imagen '{args.image}' no existe.")
            sys.exit(1)
            
        print(f"Cargando e infiriendo imagen externa: {args.image}")
        from PIL import Image
        try:
            raw_img = Image.open(args.image).convert("RGB")
        except Exception as e:
            print(f"Error al abrir la imagen '{args.image}': {e}")
            sys.exit(1)

        transform_custom = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        
        image_tensor = transform_custom(raw_img)
        
        with torch.no_grad():
            input_tensor = image_tensor.unsqueeze(0).to(DEVICE)
            outputs, _ = net(input_tensor)
            
            probs = torch.nn.functional.softmax(outputs[0], dim=0)
            top3_prob, top3_idx = torch.topk(probs, 3)
            
            predicted_idx = top3_idx[0].item()
            confidence = top3_prob[0].item() * 100

        predicted_class = CLASSES[predicted_idx]
        print(f"\nPredicción principal: {predicted_class}")
        print(f"Confianza: {confidence:.2f}%")
        
        print("\nTop 3 clases con mayor probabilidad:")
        for i in range(3):
            cls_name = CLASSES[top3_idx[i].item()]
            prob = top3_prob[i].item() * 100
            print(f"  {i+1}. {cls_name}: {prob:.2f}%")

        img_unnorm = image_tensor.clone()
        for t, m, s in zip(img_unnorm, mean, std):
            t.mul_(s).add_(m)
        img_unnorm = torch.clamp(img_unnorm, 0, 1)
        
        img_np = img_unnorm.numpy().transpose(1, 2, 0)
        
        model_name = os.path.splitext(os.path.basename(args.model))[0]
        preds_dir = os.path.join("predictions", model_name)
        os.makedirs(preds_dir, exist_ok=True)
        
        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        axes[0].imshow(raw_img)
        axes[0].set_title("Imagen Original")
        axes[0].axis("off")
        
        axes[1].imshow(img_np)
        axes[1].set_title(f"Redimensionada (32x32)\nPredicción: {predicted_class} ({confidence:.1f}%)")
        axes[1].axis("off")
        
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join(preds_dir, f"custom_pred_{timestamp_str}.png")
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        
        print(f"\nVisualización guardada como '{save_path}'")
        return
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    download_cifar = not (args.eval_all or args.eval_n)

    try:
        testset = torchvision.datasets.CIFAR10(
            root=CIFAR_DIR, train=False, download=download_cifar, transform=transform_test
        )
    except Exception as e:
        print(f"Error al cargar CIFAR-10. Si usa evaluación masiva, asegúrese de que el dataset esté descargado en {CIFAR_DIR}.")
        sys.exit(1)
    
    dataset_size = len(testset)

    # Modo predicción individual
    if args.idx is not None or args.random:
        if args.random:
            idx = random.randint(0, dataset_size - 1)
        else:
            idx = args.idx

        if idx < 0 or idx >= dataset_size:
            print(f"Error: El índice {idx} está fuera de rango. Debe ser entre 0 y {dataset_size - 1}.")
            sys.exit(1)

        print(f"Índice de CIFAR-10 utilizado: {idx}")

        image_tensor, label_idx = testset[idx]
        
        with torch.no_grad():
            input_tensor = image_tensor.unsqueeze(0).to(DEVICE)
            outputs, _ = net(input_tensor)
            
            probs = torch.nn.functional.softmax(outputs[0], dim=0)
            top3_prob, top3_idx = torch.topk(probs, 3)
            
            predicted_idx = top3_idx[0].item()
            confidence = top3_prob[0].item() * 100

        real_class = CLASSES[label_idx]
        predicted_class = CLASSES[predicted_idx]

        print(f"Clase real: {real_class}")
        print(f"Clase predicha: {predicted_class}")
        print(f"Confianza de la predicción: {confidence:.2f}%")
        
        print("Top 3 clases con mayor probabilidad:")
        for i in range(3):
            cls_name = CLASSES[top3_idx[i].item()]
            prob = top3_prob[i].item() * 100
            print(f"  {i+1}. {cls_name}: {prob:.2f}%")

        img_unnorm = image_tensor.clone()
        for t, m, s in zip(img_unnorm, mean, std):
            t.mul_(s).add_(m)
        img_unnorm = torch.clamp(img_unnorm, 0, 1)
        
        img_np = img_unnorm.numpy().transpose(1, 2, 0)
        
        model_name = os.path.splitext(os.path.basename(args.model))[0]
        preds_dir = os.path.join("predictions", model_name)
        os.makedirs(preds_dir, exist_ok=True)
        
        plt.figure(figsize=(4, 4))
        plt.imshow(img_np)
        
        title = f"Real: {real_class} | Predicción: {predicted_class} ({confidence:.1f}%)"
        plt.title(title)
        plt.axis("off")
        
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join(preds_dir, f"prediction_{idx}_{timestamp_str}.png")
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        
        print(f"Imagen guardada como '{save_path}'")

    # Modo evaluación completa
    else:
        if args.eval_all:
            eval_dataset = testset
            num_eval = dataset_size
            file_suffix = "all"
        else:
            num_eval = min(args.eval_n, dataset_size)
            rng = random.Random(args.eval_seed)
            indices = rng.sample(range(dataset_size), num_eval)
            eval_dataset = Subset(testset, indices)
            file_suffix = str(num_eval)
            
        print(f"Cantidad de imágenes a evaluar: {num_eval}")
        
        loader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False)
        
        top1_correct = 0
        top3_correct = 0
        total_confidence = 0.0
        
        class_correct = {i: 0 for i in range(10)}
        class_total = {i: 0 for i in range(10)}

        start_time = time.time()
        
        with torch.no_grad():
            for inputs, targets in loader:
                inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
                
                outputs, _ = net(inputs)
                probs = torch.nn.functional.softmax(outputs, dim=1)
                
                # Confianza max (top 1)
                max_probs, preds = torch.max(probs, dim=1)
                total_confidence += max_probs.sum().item()
                
                # Top 1 aciertos
                top1_correct += (preds == targets).sum().item()
                
                # Accuracy por clase
                for t, p in zip(targets, preds):
                    class_total[t.item()] += 1
                    if t == p:
                        class_correct[t.item()] += 1
                
                # Top 3 aciertos
                _, top3_preds = torch.topk(probs, 3, dim=1)
                for i in range(targets.size(0)):
                    if targets[i] in top3_preds[i]:
                        top3_correct += 1

        eval_time = time.time() - start_time
        img_per_sec = num_eval / eval_time if eval_time > 0 else 0
        
        top1_acc = (top1_correct / num_eval) * 100
        top3_acc = (top3_correct / num_eval) * 100
        avg_conf = (total_confidence / num_eval) * 100

        # Preparar texto de resumen
        summary = []
        summary.append(f"Ruta del modelo: {args.model}")
        summary.append(f"Canales: {checkpoint['channel']}")
        summary.append(f"Cantidad de celdas: {checkpoint['num_cells']}")
        summary.append(f"Accuracy guardada: {acc_saved if acc_saved is not None else 'No disponible'}")
        summary.append(f"Cantidad de imágenes evaluadas: {num_eval}")
        summary.append(f"Top-1 accuracy: {top1_acc:.2f}%")
        summary.append(f"Top-3 accuracy: {top3_acc:.2f}%")
        summary.append(f"Cantidad de aciertos top-1: {top1_correct}")
        summary.append(f"Cantidad de aciertos top-3: {top3_correct}")
        summary.append(f"Confianza promedio de la clase predicha: {avg_conf:.2f}%")
        summary.append(f"Tiempo total de evaluación: {eval_time:.2f} s")
        summary.append(f"Imágenes por segundo: {img_per_sec:.2f}")
        summary.append("\nAccuracy por clase:")
        
        for i in range(10):
            total_c = class_total[i]
            if total_c > 0:
                acc_c = (class_correct[i] / total_c) * 100
                summary.append(f"{CLASSES[i]}: {class_correct[i]}/{total_c} = {acc_c:.2f}%")
            else:
                summary.append(f"{CLASSES[i]}: 0/0 = 0.00%")
                
        summary_text = "\n".join(summary)
        print("\n" + summary_text)
        
        # Guardar en archivo
        model_name = os.path.splitext(os.path.basename(args.model))[0]
        preds_dir = os.path.join("predictions", model_name)
        os.makedirs(preds_dir, exist_ok=True)
        
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_name = f"eval_{file_suffix}_{timestamp_str}.txt"
        save_path = os.path.join(preds_dir, file_name)
        
        with open(save_path, "w", encoding="utf-8") as f:
            f.write(summary_text)
            
        print(f"\nResumen guardado en '{save_path}'")

if __name__ == "__main__":
    main()