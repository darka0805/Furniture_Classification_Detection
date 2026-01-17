import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import torchvision.transforms.functional as F
import numpy as np
import argparse
import os

class SquarePad:
    def __call__(self, image):
        w, h = image.size
        max_wh = max(w, h)
        hp = (max_wh - w) // 2
        vp = (max_wh - h) // 2
        padding = (hp, vp, max_wh - w - hp, max_wh - h - vp)
        return F.pad(image, padding, 0, 'constant')

def evaluate_model(data_dir, model_path="best_model.pth", batch_size=32):
    device = torch.device("cpu") 
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("GPU detected, using CUDA.")
    else:
        print("Using CPU.")

    print(f"Loading data from: {data_dir}")
    
    data_transforms = transforms.Compose([
        SquarePad(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    try:
        full_dataset = datasets.ImageFolder(data_dir, transform=data_transforms)
    except FileNotFoundError:
        print(f"Error: Data directory '{data_dir}' not found.")
        return

    classes = full_dataset.classes
    print(f"Classes: {classes}")

    targets = np.array(full_dataset.targets)
    
    train_idx, rem_idx = train_test_split(
        np.arange(len(targets)), 
        test_size=0.30, 
        random_state=42, 
        stratify=targets
    )
    
    rem_targets = targets[rem_idx]
    val_idx, test_idx = train_test_split(
        rem_idx, 
        test_size=0.50, 
        random_state=42, 
        stratify=rem_targets
    )

    test_data = Subset(full_dataset, test_idx)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    
    print(f"Test Set Size: {len(test_data)} images")

    print(f"Loading model from: {model_path}")
    if not os.path.exists(model_path):
        print("Error: Model file not found.")
        return

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(classes))
    
    try:
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
    except Exception as e:
        print(f"Error loading state_dict: {e}")
        return

    model = model.to(device)
    model.eval()

    print("\n Evaluating on Test Set...")
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    print("\n" + "="*60)
    print("CLASSIFICATION REPORT")
    print("="*60)
    print(classification_report(all_labels, all_preds, target_names=classes))
    
    print("\n" + "="*60)
    print("CONFUSION MATRIX")
    print("="*60)
    cm = confusion_matrix(all_labels, all_preds)
    print(cm)
    
    print("\n(Rows: True, Cols: Predicted)")
    row_format = "{:>15}" * (len(classes) + 1)
    print(row_format.format("", *classes))
    for class_name, row in zip(classes, cm):
        print(row_format.format(class_name, *row))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Cabinet Classifier on Test Set")
    parser.add_argument("--data_dir", type=str, default="dataset", help="Path to dataset folder containing class folders")
    parser.add_argument("--model_path", type=str, default="best_model.pth", help="Path to the saved model")
    
    args = parser.parse_args()
    
    evaluate_model(args.data_dir, args.model_path)

