import cv2
import tensorflow as tf
import numpy as np
import time
import argparse

def run_detection(model_path, image_path, classes_path, confidence_thresh, iou_thresh):
    """
    Loads a TFLite model, runs inference on an image, and saves the output.
    """
    try:
        with open(classes_path, 'r') as f:
            CLASSES = [line.strip() for line in f.readlines()]
    except FileNotFoundError:
        print(f"Error: Classes file not found at '{classes_path}'")
        return

    print("Loading TFLite model...")
    try:
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_shape = input_details[0]['shape']
    input_height, input_width = input_shape[1], input_shape[2]

    try:
        img = cv2.imread(image_path)
        if img is None:
            print(f"Error: Could not read image from '{image_path}'")
            return
            
        original_height, original_width, _ = img.shape
        img_resized = cv2.resize(img, (input_width, input_height))
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        input_data = np.expand_dims(img_rgb, axis=0)

        # Handle normalization based on model's input type
        if input_details[0]['dtype'] == np.float32:
             input_data = input_data.astype(np.float32) / 255.0

    except Exception as e:
        print(f"Error preprocessing image: {e}")
        return

    interpreter.set_tensor(input_details[0]['index'], input_data)
    print("Running inference...")
    start_time = time.time()
    interpreter.invoke()
    end_time = time.time()
    print(f"Inference time: {end_time - start_time:.4f} seconds")

    output_data = interpreter.get_tensor(output_details[0]['index'])
    
    # Transpose output from [1, 8, 2100] to [2100, 8]
    if len(output_data.shape) == 3:
        output_data = np.transpose(output_data[0], (1, 0))
    else:
        output_data = output_data[0]
    
    boxes, scores, class_ids = [], [], []

    for row in output_data:
        confidence = np.max(row[4:])
        
        if confidence > confidence_thresh:
            class_id = np.argmax(row[4:])
            scores.append(float(confidence))
            class_ids.append(int(class_id))
            
            # --- THE CORRECT LOGIC ---
            # The model output (cx, cy, w, h) is NORMALIZED by the input image dimensions.
            cx_norm, cy_norm, w_norm, h_norm = row[:4]

            # Convert normalized coordinates to pixel coordinates on the ORIGINAL image
            cx = cx_norm * original_width
            cy = cy_norm * original_height
            w = w_norm * original_width
            h = h_norm * original_height

            # Calculate top-left corner (x1, y1)
            x1 = int(cx - w / 2)
            y1 = int(cy - h / 2)
            
            # We already have width and height, just need to cast to int
            width = int(w)
            height = int(h)
            
            boxes.append([x1, y1, width, height])
            # --- END OF FIX ---

    print(f"\nDetections above confidence threshold ({confidence_thresh}): {len(boxes)}")

    if len(boxes) > 0:
        indices = cv2.dnn.NMSBoxes(boxes, scores, confidence_thresh, iou_thresh)
        print(f"Found {len(indices) if indices is not None else 0} objects after NMS.")
        
        if indices is not None and len(indices) > 0:
            indices_flat = indices.flatten()
                
            for i in indices_flat:
                x, y, w, h = boxes[i]
                class_id = class_ids[i]
                score = scores[i]
                
                class_name = CLASSES[class_id] if 0 <= class_id < len(CLASSES) else f"Unknown"
                
                # Draw rectangle and label
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                label = f"{class_name}: {score:.2f}"
                label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                label_y = y - 10 if y - 10 > 10 else y + 10
                cv2.rectangle(img, (x, label_y - label_size[1]), (x + label_size[0], label_y + 5), (0, 255, 0), cv2.FILLED)
                cv2.putText(img, label, (x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
                
                print(f"  -> Detection: {class_name} - Conf: {score:.2f} - Box: ({x},{y},{w},{h})")

    output_filename = "test_out/scene_01.jpg"
    cv2.imwrite(output_filename, img)
    print(f"\nOutput image saved as '{output_filename}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLOv8 TFLite Inference")
    parser.add_argument('--model', type=str, default='../runs/detect/helmet_detection/weights/best_saved_model/best_float32.tflite', help='Path to the TFLite model file.')
    parser.add_argument('--image', type=str, required=True, help='Path to the input image.')
    parser.add_argument('--classes', type=str, default='classes.txt', help='Path to the classes text file.')
    parser.add_argument('--conf', type=float, default=0.4, help='Confidence threshold.')
    parser.add_argument('--iou', type=float, default=0.3, help='IOU threshold for NMS.')
    args = parser.parse_args()

    run_detection(args.model, args.image, args.classes, args.conf, args.iou)