import os
from flask import Flask, request, jsonify, render_template, send_from_directory
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import io

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['RESULT_FOLDER'] = 'results/'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULT_FOLDER'], exist_ok=True)

# Load your trained YOLOv11 model
# Make sure your 'best.pt' file is in the same directory as app.py or provide the full path
try:
    model = YOLO("best.pt")
    print("YOLOv11 model loaded successfully.")
except Exception as e:
    print(f"Error loading YOLOv11 model: {e}")
    print("Please ensure 'best.pt' is in the current directory or provide the correct path.")
    exit()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'image' not in request.files:
        return jsonify({'error': 'No image part in the request'}), 400

    file = request.files['image']

    if file.filename == '':
        return jsonify({'error': 'No selected image'}), 400

    if file:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        try:
            # Open the image using PIL to ensure it's in a format YOLO can handle
            image = Image.open(filepath).convert("RGB")
            image_np = np.array(image)

            # Perform inference
            results = model(image_np)

            # Process and save the annotated image
            annotated_image = results[0].plot() # This plots bounding boxes, labels, etc.
            
            # Convert annotated_image (NumPy array) to PIL Image for saving
            annotated_image_pil = Image.fromarray(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB))
            
            result_filename = f"result_{file.filename}"
            result_filepath = os.path.join(app.config['RESULT_FOLDER'], result_filename)
            annotated_image_pil.save(result_filepath)

            # Prepare detection details
            detections = []
            for r in results[0].boxes:
                x1, y1, x2, y2 = r.xyxy[0].tolist()
                conf = r.conf[0].item()
                cls = int(r.cls[0].item())
                name = model.names[cls]
                detections.append({
                    "box": [x1, y1, x2, y2],
                    "confidence": conf,
                    "class_id": cls,
                    "class_name": name
                })

            return jsonify({
                'message': 'Image processed successfully',
                'original_image': f'/uploads/{file.filename}',
                'result_image': f'/results/{result_filename}',
                'detections': detections
            }), 200

        except Exception as e:
            print(f"Error during inference: {e}")
            return jsonify({'error': f'Error processing image: {e}'}), 500

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/results/<filename>')
def result_file(filename):
    return send_from_directory(app.config['RESULT_FOLDER'], filename)

if __name__ == '__main__':
   ## app.run(host='127.0.0.1', port=5000, debug=True)
