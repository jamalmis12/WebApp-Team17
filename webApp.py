import torch
from flask import Flask, request, render_template, send_file
import cv2
from PIL import Image
import io
import numpy as np
import math
from yolov5 import YOLOv5  # Import YOLOv5

app = Flask(__name__)

# Load the YOLOv5 model (adjust path to your actual best.pt file)
model = YOLOv5('best.pt', device='cpu')  # Use 'cuda' if you have a GPU

# Function to calculate the Cobb angle
def calculate_cobb_angle(points):
    # Points should be a list of 2 tuples: [(x1, y1), (x2, y2)]
    (x1, y1), (x2, y2) = points
    
    # Calculate the slope of the line formed by two vertebrae
    # Avoid division by zero
    if x1 - x2 != 0:
        slope = (y1 - y2) / (x1 - x2)
        angle = math.degrees(math.atan(abs(slope)))
        return angle
    return 0  # In case the line is vertical (undefined slope)

# Prediction route (for handling image uploads)
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return 'No file part', 400
    file = request.files['file']
    if file.filename == '':
        return 'No selected file', 400

    # Convert file to PIL image
    img = Image.open(file.stream)
    
    # Convert PIL image to numpy array
    img_array = np.array(img)
    img_array = img_array[..., ::-1]  # Convert RGB to BGR

    # Perform inference
    results = model.predict(img_array)  # Make predictions on the image
    
    # Get the image with bounding boxes drawn
    output_img = results.render()[0]  # This will draw the boxes on the image

    # Ensure the image is writable
    output_img = np.array(output_img, copy=True)

    # Extract the coordinates of the detected vertebrae
    detections = results.xywh[0].numpy()  # Get the bounding boxes (x, y, width, height, confidence)
    vertebrae_points = []

    # Loop through detections
    for detection in detections:
        try:
            x_center, y_center, width, height, confidence, class_id = detection  # Correct order
        except ValueError as e:
            print("Error unpacking detection:", detection, e)
            continue

        # Check if class_id corresponds to vertebrae (class_id == 0)
        if int(class_id) == 0:
            vertebrae_points.append((int(x_center), int(y_center)))

    print("Vertebrae Points:", vertebrae_points)  # Debugging: Verify points

    # Calculate Cobb angle and draw lines if enough vertebrae are detected
    cobb_angle = None
    if len(vertebrae_points) >= 3:
        # Identify apex vertebra: we assume it's the middle point for simplicity
        apex_point = vertebrae_points[len(vertebrae_points) // 2]
        
        # Identify the upper and lower vertebrae
        upper_point = vertebrae_points[0]
        lower_point = vertebrae_points[-1]
        
        # Calculate Cobb angle
        cobb_angle = calculate_cobb_angle([upper_point, lower_point])
        
        # Draw lines connecting upper and lower vertebrae
        cv2.line(output_img, upper_point, lower_point, (255, 0, 255), 2)  # Yellow line for upper-lower connection
        
        # Draw a distinct dot for the apex vertebra
        cv2.circle(output_img, apex_point, 10, (0, 255, 255), -1)  # Blue dot for apex vertebra
        
        # Draw circles for upper and lower vertebrae
        cv2.circle(output_img, upper_point, 5, (255, 255, 0), -1)  # Green circle for upper vertebra
        cv2.circle(output_img, lower_point, 5, (0, 0, 255), -1)  # Red circle for lower vertebra

        # Annotate the Cobb angle only if it's calculated
        if cobb_angle is not None and cobb_angle != 0:  # Ensure a valid angle is displayed
            angle_text = f'Cobb Angle: {cobb_angle:.2f}°'
            # Positioning it closer to the right side of the image
            text_size = cv2.getTextSize(angle_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            x_position = output_img.shape[1] - text_size[0] - 15  # 10px margin from the right edge
            y_position = upper_point[1] - 10  # Adjusted y-position to be slightly above the upper point
            cv2.putText(output_img, angle_text, (x_position, y_position),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)  # White text
        else:
            angle_text = "Cobb Angle: Invalid"
            # Positioning it closer to the right side of the image
            text_size = cv2.getTextSize(angle_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            x_position = output_img.shape[1] - text_size[0] - 10  # 10px margin from the right edge
            y_position = upper_point[1] - 10  # Adjusted y-position to be slightly above the upper point
            cv2.putText(output_img, angle_text, (x_position, y_position),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)  # White text
    else:
        print("Not enough vertebrae detected to calculate Cobb angle.")
        angle_text = "Not enough vertebrae detected"
        # Positioning it closer to the right side of the image
        text_size = cv2.getTextSize(angle_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
        x_position = output_img.shape[1] - text_size[0] - 10  # 10px margin from the right edge
        y_position = 30  # Place it at the top of the image
        cv2.putText(output_img, angle_text, (x_position, y_position),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)  # White text

    # Convert the processed image to PIL object
    output_pil = Image.fromarray(output_img)
    
    # Save the processed image to a BytesIO object
    img_io = io.BytesIO()
    output_pil.save(img_io, 'PNG')
    img_io.seek(0)

    # Return processed image
    return send_file(img_io, mimetype='image/png', as_attachment=True, download_name='processed_image.png')

# Home route
@app.route('/')
def index():
    return render_template('index.html')  # Ensure you have an index.html file in templates

if __name__ == '__main__':
    app.run(debug=True)
