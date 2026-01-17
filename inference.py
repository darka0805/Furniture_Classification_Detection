import torch
import torch.nn as nn
from torchvision import models, transforms
import torchvision.transforms.functional as F
from PIL import Image
import argparse
import os
import sys

CLASSES = [
    'lc_bcabo',
    'lc_bcabocub', 
    'lc_muscabinso', 
    'lc_wcabcub', 
    'lc_wcabo'
]

class SquarePad:
    def __call__(self, image):
        w, h = image.size
        max_wh = max(w, h)
        hp = (max_wh - w) // 2
        vp = (max_wh - h) // 2
        padding = (hp, vp, max_wh - w - hp, max_wh - h - vp)
        return F.pad(image, padding, 0, 'constant')

def get_model(num_classes, device):
    model = models.resnet18(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    model = model.to(device)
    return model

def load_checkpoint(model, checkpoint_path, device):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
    
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model

def predict_image(model, image_path, device, transform):
    try:
        image = Image.open(image_path).convert('RGB')
        image = transform(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            outputs = model(image)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted_idx = torch.max(probs, 1)
            
        predicted_class = CLASSES[predicted_idx.item()]
        return predicted_class, confidence.item()
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None, None

def main():
    parser = argparse.ArgumentParser(description="Cabinet Classification Inference")
    parser.add_argument("input_path", type=str, help="Path to an image or directory of images")
    parser.add_argument("--model_path", type=str, default="best_model.pth", help="Path to the trained model checkpoint")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = get_model(len(CLASSES), device)
    
    try:
        model = load_checkpoint(model, args.model_path, device)
        print(f"Model loaded from {args.model_path}")
    except Exception as e:
        print(f"Failed to load model: {e}")
        sys.exit(1)

    transform = transforms.Compose([
        SquarePad(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    if os.path.isdir(args.input_path):
        print(f"\nProcessing directory: {args.input_path}")
        results = []
        for filename in os.listdir(args.input_path):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                file_path = os.path.join(args.input_path, filename)
                pred_class, conf = predict_image(model, file_path, device, transform)
                if pred_class:
                    print(f"{filename}: {pred_class} ({conf:.2f})")
                    results.append((filename, pred_class, conf))
    elif os.path.isfile(args.input_path):
        print(f"\nProcessing file: {args.input_path}")
        pred_class, conf = predict_image(model, args.input_path, device, transform)
        if pred_class:
            print(f"Prediction: {pred_class}")
            print(f"Confidence: {conf:.4f}")
    else:
        print("Invalid input path")

if __name__ == "__main__":
    main()
