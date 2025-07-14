import os
from ultralytics import YOLO

# --- Configuration ---
# Path to the data.yaml file created by download.py
DATA_YAML_PATH = "./datasets/combined_dataset/data.yaml"

# Training parameters
EPOCHS = 100    # Updated to 100 epochs
PATIENCE = 5    # Added patience for early stopping
IMG_SIZE = 320  # Smaller size for better performance on Raspberry Pi
BATCH_SIZE = 16 # Adjust based on your RunPod's VRAM
MODEL_NAME = 'helmet_detection' # Updated name for the new run

def train_and_export_model():
    """
    Trains a YOLOv8n model on the prepared dataset and exports it
    to TFLite format for Raspberry Pi deployment.
    """
    # --- Step 1: Verify Dataset ---
    if not os.path.exists(DATA_YAML_PATH):
        print(f"Error: Data YAML file not found at '{DATA_YAML_PATH}'")
        print("Please run 'download.py' first to prepare the dataset.")
        return

    print("--- Step 1: Starting YOLOv8n Training ---")

    try:
        # Load a pre-trained YOLOv8n model to start from
        model = YOLO('yolov8n.pt')

        # Train the model using the prepared dataset
        results = model.train(
            data=DATA_YAML_PATH,
            epochs=EPOCHS,
            patience=PATIENCE, # Added early stopping patience
            imgsz=IMG_SIZE,
            batch=BATCH_SIZE,
            name=MODEL_NAME
        )

        print("\n--- Training Finished ---")
        print(f"Model and results saved in the 'runs/detect/{MODEL_NAME}' directory.")

    except Exception as e:
        print(f"\nAn error occurred during training: {e}")
        return

    # --- Step 2: Export the Model for Raspberry Pi ---
    print("\n--- Step 2: Exporting Model to TFLite ---")
    try:
        # The 'train' method returns a results object containing the path to the best model
        best_model_path = results.save_dir / 'weights' / 'best.pt'

        if not os.path.exists(best_model_path):
            print(f"Error: Could not find the best model at {best_model_path}")
            return

        print(f"Loading best model from: {best_model_path}")
        model_to_export = YOLO(best_model_path)

        # Export the model to TFLite format
        # The exported file will be saved in the same directory as the .pt file
        model_to_export.export(format='tflite')

        tflite_path = results.save_dir / 'weights' / 'best.tflite'
        print("\n--- Export Complete ---")
        print(f"TFLite model saved successfully at: {tflite_path}")
        print("You can now download this .tflite file and transfer it to your Raspberry Pi.")

    except Exception as e:
        print(f"\nAn error occurred during model export: {e}")


if __name__ == "__main__":
    train_and_export_model()
