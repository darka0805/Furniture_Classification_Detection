from ultralytics import YOLO
import argparse

def evaluate_yolo(model_path, data_yaml, imgsz):
    model = YOLO(model_path)

    test_results = model.val(
        data=data_yaml, 
        split='test',
        imgsz=imgsz
    )

    print("FINAL TEST METRICS")
    print(f"mAP50: {test_results.box.map50:.4f}")
    print(f"mAP50-95: {test_results.box.map:.4f}")
    print(f"Precision: {test_results.box.mp:.4f}")
    print(f"Recall: {test_results.box.mr:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate YOLOv8 on Test Set")
    parser.add_argument("--model_path", type=str, required=True, help="Path to best.pt")
    parser.add_argument("--data", type=str, default="dataset.yaml", help="Path to dataset.yaml")
    parser.add_argument("--imgsz", type=int, default=1024, help="Image size")

    args = parser.parse_args()
    
    evaluate_yolo(args.model_path, args.data, args.imgsz)
