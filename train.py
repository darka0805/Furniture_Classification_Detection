import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
import torchvision.transforms.functional as F
import numpy as np
import argparse
import os
import time

class SquarePad:
    def __call__(self, image):
        w, h = image.size
        max_wh = max(w, h)
        hp = (max_wh - w) // 2
        vp = (max_wh - h) // 2
        padding = (hp, vp, max_wh - w - hp, max_wh - h - vp)
        return F.pad(image, padding, 0, 'constant')

def train_model(data_dir, output_model_path="best_model.pth", epochs=15, batch_size=32, learning_rate=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

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

    train_data = Subset(full_dataset, train_idx)
    val_data = Subset(full_dataset, val_idx)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    
    print(f"Data Split: Train={len(train_data)}, Val={len(val_data)}, Test={len(test_idx)}")

    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, len(classes))
    model = model.to(device)

    class_counts = [len(os.listdir(os.path.join(data_dir, d))) for d in classes]
    weights = 1. / torch.tensor(class_counts, dtype=torch.float)
    normalized_weights = (weights / weights.sum() * len(class_counts)).to(device)
    
    criterion = nn.CrossEntropyLoss(weight=normalized_weights)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    best_val_loss = float('inf')
    
    print("Starting training...")
    start_time = time.time()

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
        
        train_loss = running_loss / len(train_data)

        model.eval()
        val_running_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_running_loss += loss.item() * inputs.size(0)
        
        val_loss = val_running_loss / len(val_data)

        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), output_model_path)
            print(f"  --> Saved new best model to {output_model_path}")

    total_time = time.time() - start_time
    print(f"Training complete in {total_time // 60:.0f}m {total_time % 60:.0f}s")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Cabinet Classifier")
    parser.add_argument("--data_dir", type=str, default="dataset", help="Path to dataset folder containing class folders")
    parser.add_argument("--output", type=str, default="best_model.pth", help="Path to save the best model")
    parser.add_argument("--epochs", type=int, default=15, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    
    args = parser.parse_args()
    
    train_model(args.data_dir, args.output, args.epochs, args.batch_size, args.lr)
