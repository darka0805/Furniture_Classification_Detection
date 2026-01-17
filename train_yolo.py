from ultralytics import YOLO
import argparse
import os

def train_yolo(data_yaml, epochs=100, batch=8, imgsz=1024, model_name='yolov8n.pt', project_name='cabinet_detector'):
    print(f"Starting training with:")
    print(f"Data: {data_yaml}")
    print(f"Epochs: {epochs}")
    print(f"Batch: {batch}")
    print(f"Image Size: {imgsz}")
    
    model = YOLO(model_name)

    model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        name=project_name,
        augment=True,
        patience=20,
        device=0 if os.path.exists("cuda") else 'cpu'
    )
    
    return model

def validate_model(model_path, data_yaml, imgsz=1024):
    print(f"\nValidating Model {model_path}")
    
    model = YOLO(model_path)
    
    results = model.val(
        data=data_yaml, 
        split='test',
        imgsz=imgsz
    )

    print("FINAL TEST METRICS")
    print(f"mAP50: {results.box.map50:.4f}")
    print(f"mAP50-95: {results.box.map:.4f}")
    print(f"Precision: {results.box.mp:.4f}")
    print(f"Recall: {results.box.mr:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train YOLOv8 on Cabinet Data")
    parser.add_argument("--data", type=str, default="dataset.yaml", help="Path to dataset.yaml")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=8, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=1024, help="Image size")
    parser.add_argument("--weights", type=str, default="yolov8n.pt", help="Initial weights")
    parser.add_argument("--name", type=str, default="balanced_cabinet_detector", help="Project run name")
    parser.add_argument("--validate_only", action="store_true", help="Skip training and just validate existing model")
    parser.add_argument("--model_path", type=str, help="Path to best.pt for validation (required if validate_only)")

    args = parser.parse_args()

    if args.validate_only:
        if not args.model_path:
            print("Error: --model_path is required for validation only mode.")
        else:
            validate_model(args.model_path, args.data, args.imgsz)
    else:
        model = train_yolo(args.data, args.epochs, args.batch, args.imgsz, args.weights, args.name)
        best_path = os.path.join("runs", "detect", args.name, "weights", "best.pt")
        if os.path.exists(best_path):
            validate_model(best_path, args.data, args.imgsz)
        else:
            print(f"Could not find best model at {best_path} to validate. Please check output directory.")
