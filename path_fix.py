import zipfile
import os
import json

def extract_and_fix_zip(zip_path, extract_to_folder):
    """
    Extracts a zip file while renaming files to be suatable for Windows system
    """
    if not os.path.exists(extract_to_folder):
        os.makedirs(extract_to_folder)
    
    with zipfile.ZipFile(zip_path, 'r') as z:
        all_files = z.namelist()
        rename_map = {}

        for file_info in z.infolist():
            original_path = file_info.filename
            new_relative_path = original_path.replace(":", "_")
            rename_map[original_path] = new_relative_path
            target_path = os.path.join(extract_to_folder, new_relative_path)
            os.makedirs(os.path.dirname(target_path), exist_ok=True)
            
            if not file_info.is_dir():
                with z.open(original_path) as source, open(target_path, "wb") as target:
                    target.write(source.read())

    for subdir, dirs, files in os.walk(extract_to_folder):
        if "simple_annotations.json" in files:
            json_path = os.path.join(subdir, "simple_annotations.json")
            try:
                with open(json_path, 'r') as f:
                    data = json.load(f)
                
                updated = False
                if 'images' in data:
                    for img in data['images']:
                        if ":" in img['file_name']:
                            img['file_name'] = img['file_name'].replace(":", "_")
                            updated = True
                
                if updated:
                    with open(json_path, 'w') as f:
                        json.dump(data, f, indent=4)
                    print(f"Updated JSON: {json_path}")
            except Exception as e:
                print(f"Could not update JSON {json_path}: {e}")

def clean_filenames(root_folder):
    print(f"Cleaning filenames in: {root_folder}")
    count = 0
    for root, dirs, files in os.walk(root_folder):
        for filename in files:
            if "[" in filename or "]" in filename:
                new_name = filename.replace("[", "").replace("]", "").replace(":", "_")
                
                old_path = os.path.join(root, filename)
                new_path = os.path.join(root, new_name)
                
                try:
                    os.rename(old_path, new_path)
                    count += 1
                except OSError as e:
                    print(f"Error renaming {filename}: {e}")

    print(f"Finished! Renamed {count} files.")


if __name__ == "__main__":
    zip_path = r".\annotated_pdfs_and_data.zip"
    output_folder = r".\extracted_data_clean"
    extract_and_fix_zip(zip_path, output_folder)
    clean_filenames(output_folder)