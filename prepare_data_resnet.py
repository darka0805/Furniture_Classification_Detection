import os
import json
import glob
from PIL import Image
from tqdm import tqdm
import shutil

DATA_ROOT = r".\extracted_data_clean"
OUTPUT_DIR = r".\classifier\dataset"

TARGET_CATEGORIES = {
    "lc:bcabo",
    "lc:wcabo",
    "lc:muscabinso",
    "lc:wcabcub",
    "lc:bcabocub"
}

def clean_name(name):
    return name.replace(":", "_")

def prepare_data():
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    
    for cat in TARGET_CATEGORIES:
        os.makedirs(os.path.join(OUTPUT_DIR, clean_name(cat)), exist_ok=True)

    json_files = []
    print("Scanning for JSON files...")
    for root, dirs, files in os.walk(DATA_ROOT):
        for file in files:
            if file.endswith("_simple.json"):
                json_files.append(os.path.join(root, file))

    print(f"Found {len(json_files)} JSON files.")

    count = 0
    errors = 0
    
    for json_file in tqdm(json_files):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            image_map = {img['id']: img['file_name'] for img in data.get('images', [])}
            
            for ann in data.get('annotations', []):
                label = ann.get('label')
                
                if label not in TARGET_CATEGORIES:
                    continue
                
                image_id = ann.get('image_id')
                if image_id not in image_map:
                    continue
                
                image_filename = image_map[image_id]
                image_path = os.path.join(os.path.dirname(json_file), image_filename)
                
                if not os.path.exists(image_path):
                    pass
                    continue

                try:
                    with Image.open(image_path) as img:
                        bbox = ann.get('bbox')
                        if not bbox:
                            continue
                        
                        x, y, w, h = bbox
                        x, y, w, h = int(x), int(y), int(w), int(h)
                        
                        img_w, img_h = img.size
                        x = max(0, x)
                        y = max(0, y)
                        w = min(w, img_w - x)
                        h = min(h, img_h - y)
                        
                        if w <= 0 or h <= 0:
                            continue

                        crop = img.crop((x, y, x + w, y + h))
                        
                        parent_folder = os.path.basename(os.path.dirname(json_file))
                        base_name = os.path.splitext(image_filename)[0]
                        out_filename = f"{parent_folder}_{base_name}_{ann['id']}.png"
                        out_path = os.path.join(OUTPUT_DIR, clean_name(label), out_filename)
                        
                        crop.save(out_path)
                        count += 1
                except Exception as e:
                    print(f"Error processing image {image_path}: {e}")
                    errors += 1
                    
        except Exception as e:
            print(f"Error processing file {json_file}: {e}")
            errors += 1

    print(f"Processed {count} images. Errors: {errors}")
    
    print("\nDataset Statistics:")
    for cat in TARGET_CATEGORIES:
        safe_cat = clean_name(cat)
        if os.path.exists(os.path.join(OUTPUT_DIR, safe_cat)):
            num_files = len(os.listdir(os.path.join(OUTPUT_DIR, safe_cat)))
            print(f"{cat} ({safe_cat}): {num_files}")
        else:
            print(f"{cat} ({safe_cat}): 0")

if __name__ == "__main__":
    prepare_data()
