import os
import yaml
import shutil
from roboflow import Roboflow

# --- Configuration ---
# Your Roboflow API key. It's better to use environment variables in production,
# but this is fine for a personal script.
ROBOFLOW_API_KEY = "dDQkzHDAwlS6Dw9vjk0J"

# The Roboflow projects you want to download
DATASET_URLS = {
    "motobike": "cdio-zmfmj/motobike-detection",
    "helmet_license": "cdio-zmfmj/helmet-lincense-plate-detection-gevlq"
}

# Directory to store all datasets
WORKSPACE_DIR = "./datasets"
# Directory for the final, merged dataset
COMBINED_DIR = os.path.join(WORKSPACE_DIR, "combined_dataset")

def download_and_prepare_datasets():
    """
    Downloads datasets from Roboflow, merges them into a single dataset,
    and creates a data.yaml file for YOLOv8 training.
    """
    print("--- Step 1: Downloading Datasets ---")
    if not os.path.exists(WORKSPACE_DIR):
        os.makedirs(WORKSPACE_DIR)

    try:
        rf = Roboflow(api_key=ROBOFLOW_API_KEY)
    except Exception as e:
        print(f"Error initializing Roboflow. Check your API key. Error: {e}")
        return

    downloaded_paths = {}

    for name, url in DATASET_URLS.items():
        try:
            project = rf.workspace(url.split('/')[0]).project(url.split('/')[1])
            # Get the latest version of the dataset in yolov8 format
            version_number = project.versions()[-1].version
            print(f"Downloading version {version_number} of '{name}'...")
            dataset = project.version(version_number).download("yolov8")
            downloaded_paths[name] = dataset.location
            print(f"'{name}' dataset downloaded to: {dataset.location}")
        except Exception as e:
            print(f"Could not download dataset '{name}'. Error: {e}")
            print("Please ensure the dataset URL is correct and your API key has access.")
            continue

    if not downloaded_paths:
        print("\nNo datasets were downloaded. Exiting.")
        return

    print("\n--- Step 2: Merging Datasets ---")
    # Create subdirectories for the combined dataset
    for split in ["train", "valid", "test"]:
        os.makedirs(os.path.join(COMBINED_DIR, "images", split), exist_ok=True)
        os.makedirs(os.path.join(COMBINED_DIR, "labels", split), exist_ok=True)

    all_classes = set()
    class_map = {}

    # Process each downloaded dataset
    for name, path in downloaded_paths.items():
        print(f"Processing and merging '{name}'...")
        data_yaml_path = os.path.join(path, "data.yaml")
        if not os.path.exists(data_yaml_path):
            print(f"Warning: data.yaml not found for '{name}'. Skipping.")
            continue

        with open(data_yaml_path, 'r') as f:
            data_config = yaml.safe_load(f)
            current_classes = data_config['names']

            # Create a mapping from this dataset's class index to the new unified index
            temp_class_map = {}
            for i, class_name in enumerate(current_classes):
                if class_name not in all_classes:
                    all_classes.add(class_name)
                    class_map[class_name] = len(all_classes) - 1
                temp_class_map[i] = class_map[class_name]

            # Copy files and update label indices
            for split in ["train", "valid", "test"]:
                img_source_dir = os.path.join(path, split, "images")
                lbl_source_dir = os.path.join(path, split, "labels")

                if not os.path.exists(img_source_dir):
                    continue

                for img_file in os.listdir(img_source_dir):
                    base_name, _ = os.path.splitext(img_file)
                    lbl_file = f"{base_name}.txt"

                    # Copy image file
                    shutil.copy(os.path.join(img_source_dir, img_file), os.path.join(COMBINED_DIR, "images", split))

                    # Read, update, and write the corresponding label file
                    if os.path.exists(os.path.join(lbl_source_dir, lbl_file)):
                        with open(os.path.join(lbl_source_dir, lbl_file), 'r') as f_in, \
                             open(os.path.join(COMBINED_DIR, "labels", split, lbl_file), 'w') as f_out:
                            for line in f_in:
                                parts = line.strip().split()
                                old_class_idx = int(parts[0])
                                new_class_idx = temp_class_map[old_class_idx]
                                new_line = f"{new_class_idx} {' '.join(parts[1:])}\n"
                                f_out.write(new_line)

    print("\n--- Step 3: Creating Final YAML Configuration ---")
    # Create the final data.yaml for training
    final_class_list = sorted(list(all_classes), key=lambda x: class_map[x])

    final_yaml_content = {
        'path': os.path.abspath(COMBINED_DIR),
        'train': 'images/train',
        'val': 'images/valid',
        'test': 'images/test',
        'names': {i: name for i, name in enumerate(final_class_list)}
    }

    final_yaml_path = os.path.join(COMBINED_DIR, "data.yaml")
    with open(final_yaml_path, 'w') as f:
        yaml.dump(final_yaml_content, f, sort_keys=False, indent=4)

    print("\n--- Dataset Preparation Complete ---")
    print(f"Unified dataset is ready for training in: {COMBINED_DIR}")
    print(f"Training configuration file created at: {final_yaml_path}")
    print(f"Unified Classes: {final_class_list}")

if __name__ == "__main__":
    download_and_prepare_datasets()
