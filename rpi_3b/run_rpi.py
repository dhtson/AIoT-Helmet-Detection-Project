import cv2
import numpy as np
import time
import argparse
# Use tflite_runtime for Raspberry Pi
from tflite_runtime.interpreter import Interpreter

def run_live_detection(model_path, classes_path, camera_id, confidence_thresh, iou_thresh):
    """
    Runs live object detection on a Raspberry Pi using a TFLite model and a camera.
    """
    # ---- 1. SETUP ----
    
    # Load class names
    try:
        with open(classes_path, 'r') as f:
            CLASSES = [line.strip() for line in f.readlines()]
        print(f"Loaded {len(CLASSES)} classes: {CLASSES}")
    except FileNotFoundError:
        print(f"Error: Classes file not found at '{classes_path}'")
        return

    # Load the TFLite model and allocate tensors
    print("Loading TFLite model...")
    try:
        # Use the specialized tflite_runtime Interpreter
        interpreter = Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Get model input and output details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_shape = input_details[0]['shape']
    input_height, input_width = input_shape[1], input_shape[2]

    # Initialize the camera
    print(f"Initializing camera (ID: {camera_id})...")
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print(f"Error: Could not open camera with ID {camera_id}")
        return

    # Variables for FPS calculation
    frame_count = 0
    start_time = time.time()

    # ---- 2. LIVE DETECTION LOOP ----

    while True:
        # Read a frame from the camera
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break

        original_height, original_width, _ = frame.shape

        # Pre-process the frame
        img_resized = cv2.resize(frame, (input_width, input_height))
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        input_data = np.expand_dims(img_rgb, axis=0)
        
        # Normalize if the model is float32
        if input_details[0]['dtype'] == np.float32:
            input_data = input_data.astype(np.float32) / 255.0

        # Run inference
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        
        # Transpose output from [1, N, 8400] to [8400, N]
        if len(output_data.shape) == 3:
            output_data = np.transpose(output_data[0], (1, 0))
        else:
            output_data = output_data[0]

        # Post-process the output
        boxes, scores, class_ids = [], [], []
        for row in output_data:
            confidence = np.max(row[4:])
            if confidence > confidence_thresh:
                class_id = np.argmax(row[4:])
                scores.append(float(confidence))
                class_ids.append(int(class_id))
                
                cx_norm, cy_norm, w_norm, h_norm = row[:4]
                cx = cx_norm * original_width
                cy = cy_norm * original_height
                w = w_norm * original_width
                h = h_norm * original_height
                
                x1 = int(cx - w / 2)
                y1 = int(cy - h / 2)
                width = int(w)
                height = int(h)
                
                boxes.append([x1, y1, width, height])

        # Apply Non-Maximal Suppression
        indices = cv2.dnn.NMSBoxes(boxes, scores, confidence_thresh, iou_thresh)
        
        # Draw the final bounding boxes
        if indices is not None and len(indices) > 0:
            for i in indices.flatten():
                x, y, w, h = boxes[i]
                class_id = class_ids[i]
                score = scores[i]
                class_name = CLASSES[class_id]

                # if class_name not in ["helmet", "no helmet"]:
                #    continue

                # Draw rectangle and label
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                label = f"{class_name}: {score:.2f}"
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Calculate and display FPS
        frame_count += 1
        elapsed_time = time.time() - start_time
        if elapsed_time > 1:
            fps = frame_count / elapsed_time
            cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            # Reset counter
            frame_count = 0
            start_time = time.time()

        # Display the resulting frame
        cv2.imshow('YOLOv8 Live Detection (Press "q" to quit)', frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # ---- 3. CLEANUP ----
    cap.release()
    cv2.destroyAllWindows()
    print("Camera released and windows closed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLOv8 TFLite Live Detection on Raspberry Pi")
    parser.add_argument('--model', type=str, default='../runs/detect/helmet_detection/weights/best_saved_model/best_float32.tflite', help='Path to the TFLite model file.')
    parser.add_argument('--classes', type=str, default='classes.txt', help='Path to the classes text file.')
    parser.add_argument('--camera', type=int, default=0, help='ID of the camera to use (usually 0).')
    parser.add_argument('--conf', type=float, default=0.5, help='Confidence threshold.')
    parser.add_argument('--iou', type=float, default=0.4, help='IOU threshold for NMS.')
    args = parser.parse_args()

    run_live_detection(args.model, args.classes, args.camera, args.conf, args.iou)