import json
import os
import shutil
import cv2
import random
import argparse
from glob import glob
from tqdm import tqdm
from sklearn.model_selection import train_test_split

DEFAULT_SOURCE_ROOT = "dataset/extracted_data_clean" 
DEFAULT_OUTPUT_DIR = "dataset/yolo_dataset"
DEFAULT_BALANCED_PATH = "dataset/balanced_yolo"

ID_MAPPING = {
    16: 0,
    29: 1,
    94: 2,
    38: 3,
    17: 4
}

RARE_CLASS_IDS = [3, 4]
OVERSAMPLE_FACTOR = 8

def clean_name(name):
    return name.replace("[", "").replace("]", "").replace(":", "_").replace(" ", "_")

def convert_to_yolo(bbox, img_w, img_h):
    x, y, w, h = bbox
    cx = (x + (w / 2)) / img_w
    cy = (y + (h / 2)) / img_h
    nw = w / img_w
    nh = h / img_h
    return cx, cy, nw, nh

def extract_and_convert(source_root, output_dir):
    print(f"Extracting and Converting from {source_root}")
    
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    
    for split in ['train', 'val']:
        os.makedirs(os.path.join(output_dir, 'images', split), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'labels', split), exist_ok=True)

    print(f"Searching for data...")
    all_pairs = []
    
    for root, dirs, files in os.walk(source_root):
        json_files = [f for f in files if f.endswith('_simple.json')]
        
        for json_file in json_files:
            json_path = os.path.join(root, json_file)
            img_name = json_file.replace('_simple.json', '.png')
            img_path = os.path.join(root, img_name)
            
            if os.path.exists(img_path):
                all_pairs.append((img_path, json_path))

    print(f"Found {len(all_pairs)} valid image-annotation pairs.")
    if len(all_pairs) == 0:
        print("Error: No data found. Check your source path.")
        return False

    train_pairs, val_pairs = train_test_split(all_pairs, test_size=0.2, random_state=42)

    def process_split(pairs, split_name):
        for img_path, json_path in tqdm(pairs, desc=f"Processing {split_name}"):
            with open(json_path, 'r') as f:
                data = json.load(f)
            
            img = cv2.imread(img_path)
            if img is None: continue
            h, w, _ = img.shape
            
            safe_base = clean_name(os.path.basename(img_path).replace('.png', ''))
            safe_img_name = f"{safe_base}.png"
            safe_txt_name = f"{safe_base}.txt"
            
            yolo_lines = []
            annotations = data.get('annotations', []) if isinstance(data, dict) else []
            
            for ann in annotations:
                cat_id = ann.get('category_id')
                if cat_id in ID_MAPPING:
                    y_id = ID_MAPPING[cat_id]
                    bbox = ann.get('bbox')
                    cx, cy, nw, nh = convert_to_yolo(bbox, w, h)
                    yolo_lines.append(f"{y_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
            
            if yolo_lines:
                shutil.copy(img_path, os.path.join(output_dir, 'images', split_name, safe_img_name))
                with open(os.path.join(output_dir, 'labels', split_name, safe_txt_name), 'w') as f:
                    f.write("\n".join(yolo_lines))

    process_split(train_pairs, 'train')
    process_split(val_pairs, 'val')
    return True

def balance_and_create_final_split(source_yolo_dir, balanced_dir):
    print(f"\nBalancing and creating final Train/Val/Test split in {balanced_dir}")
    
    if os.path.exists(balanced_dir):
        shutil.rmtree(balanced_dir)

    for split in ['train', 'val', 'test']:
        for folder in ['images', 'labels']:
            os.makedirs(os.path.join(balanced_dir, folder, split), exist_ok=True)

    all_images = glob(os.path.join(source_yolo_dir, "images", "*", "*.png"))
    
    rare_images = []
    common_images = []

    for img_path in all_images:
        label_path = img_path.replace("images", "labels").replace(".png", ".txt")
        if not os.path.exists(label_path): continue
        
        with open(label_path, 'r') as f:
            content = f.read()
            if any(f"{rid} " in content for rid in RARE_CLASS_IDS):
                rare_images.append(img_path)
            else:
                common_images.append(img_path)

    random.shuffle(rare_images)
    random.shuffle(common_images)
    
    test_count_rare = int(len(rare_images) * 0.15)
    test_count_common = int(len(common_images) * 0.15)
    
    test_set = rare_images[:test_count_rare] + common_images[:test_count_common]
    remaining_rare = rare_images[test_count_rare:]
    remaining_common = common_images[test_count_common:]

    val_count_rare = int(len(remaining_rare) * 0.176) 
    val_count_common = int(len(remaining_common) * 0.176)
    
    val_set = remaining_rare[:val_count_rare] + remaining_common[:val_count_common]
    train_set = remaining_rare[val_count_rare:] + remaining_common[val_count_common:]

    def copy_files(file_list, split, oversample=1):
        for img_path in file_list:
            lbl_path = img_path.replace("images", "labels").replace(".png", ".txt")
            base_name = os.path.basename(img_path).replace(".png", "")
            
            for i in range(oversample):
                suffix = f"_v{i}" if oversample > 1 else ""
                new_img_name = f"{base_name}{suffix}.png"
                new_lbl_name = f"{base_name}{suffix}.txt"
                
                shutil.copy(img_path, os.path.join(balanced_dir, "images", split, new_img_name))
                shutil.copy(lbl_path, os.path.join(balanced_dir, "labels", split, new_lbl_name))

    copy_files(test_set, 'test')
    copy_files(val_set, 'val')
    
    for img in train_set:
        factor = OVERSAMPLE_FACTOR if img in rare_images else 1
        copy_files([img], 'train', oversample=factor)

    print(f"Balanced Dataset Created!")
    print(f"Train: {len(os.listdir(os.path.join(balanced_dir, 'images', 'train')))} (Oversampled)")
    print(f"Val: {len(val_set)} | Test: {len(test_set)}")
    
    return balanced_dir

def create_yaml(balanced_dir, output_yaml_path):
    yaml_content = f"""
path: {os.path.abspath(balanced_dir)}
train: images/train
val: images/val
test: images/test 

names:
  0: lc_bcabo
  1: lc_wcabo
  2: lc_muscabinso
  3: lc_wcabcub
  4: lc_bcabocub
"""
    with open(output_yaml_path, 'w') as f:
        f.write(yaml_content)
    print(f"Created YAML config at: {output_yaml_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare and Balance YOLO Dataset")
    parser.add_argument("--source", default=DEFAULT_SOURCE_ROOT, help="Path to raw source data with json/png")
    parser.add_argument("--output_base", default=DEFAULT_OUTPUT_DIR, help="Path for temporary intermediate conversion")
    parser.add_argument("--balanced_output", default=DEFAULT_BALANCED_PATH, help="Path for final balanced dataset")
    parser.add_argument("--yaml_path", default="dataset.yaml", help="Output path for dataset.yaml")
    
    args = parser.parse_args()
    
    if extract_and_convert(args.source, args.output_base):
        balance_and_create_final_split(args.output_base, args.balanced_output)
        create_yaml(args.balanced_output, args.yaml_path)

