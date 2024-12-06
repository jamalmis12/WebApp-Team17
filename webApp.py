import functions_framework
from flask import Flask, request, render_template, send_file
import torch
import cv2
from PIL import Image
import io
import numpy as np
import math
from yolov5 import YOLOv5  # Import YOLOv5

app = Flask(__name__)

# Load the YOLOv5 model
model = YOLOv5('best.pt', device='cpu')  # Use 'cuda' if you have a GPU

# Function to calculate Cobb angle
def calculate_cobb_angle(points):
    (x1, y1), (x2, y2) = points
    if x1 - x2 != 0:
        slope = (y1 - y2) / (x1 - x2)
        angle = math.degrees(math.atan(abs(slope)))
        return angle
    return 0  # In case the line is vertical (undefined slope)

@functions_framework.http
def predict(request):
    # The same logic as your current predict route
    if 'file' not in request.files:
        return 'No file part', 400
    file = request.files['file']
    if file.filename == '':
        return 'No selected file', 400

    img = Image.open(file.stream)
    img_array = np.array(img)
    img_array = img_array[..., ::-1]  # Convert RGB to BGR

    results = model.predict(img_array)

    output_img = results.render()[0]
    output_img = np.array(output_img, copy=True)

    detections = results.xywh[0].numpy()
    vertebrae_points = []
    for detection in detections:
        try:
            x_center, y_center, width, height, confidence, class_id = detection
        except ValueError as e:
            print("Error unpacking detection:", detection, e)
            continue

        if int(class_id) == 0:
            vertebrae_points.append((int(x_center), int(y_center)))

    cobb_angle = None
    if len(vertebrae_points) >= 3:
        apex_point = vertebrae_points[len(vertebrae_points) // 2]
        upper_point = vertebrae_points[0]
        lower_point = vertebrae_points[-1]

        cobb_angle = calculate_cobb_angle([upper_point, lower_point])
        cv2.line(output_img, upper_point, lower_point, (255, 0, 255), 2)
        cv2.circle(output_img, apex_point, 10, (0, 255, 255), -1)
        cv2.circle(output_img, upper_point, 5, (255, 255, 0), -1)
        cv2.circle(output_img, lower_point, 5, (0, 0, 255), -1)

        if cobb_angle is not None and cobb_angle != 0:
            angle_text = f'Cobb Angle: {cobb_angle:.2f}°'
            text_size = cv2.getTextSize(angle_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            x_position = output_img.shape[1] - text_size[0] - 15
            y_position = upper_point[1] - 10
            cv2.putText(output_img, angle_text, (x_position, y_position),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        else:
            angle_text = "Cobb Angle: Invalid"
            text_size = cv2.getTextSize(angle_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            x_position = output_img.shape[1] - text_size[0] - 10
            y_position = upper_point[1] - 10
            cv2.putText(output_img, angle_text, (x_position, y_position),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    else:
        angle_text = "Not enough vertebrae detected"
        text_size = cv2.getTextSize(angle_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
        x_position = output_img.shape[1] - text_size[0] - 10
        y_position = 30
        cv2.putText(output_img, angle_text, (x_position, y_position),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    output_pil = Image.fromarray(output_img)
    img_io = io.BytesIO()
    output_pil.save(img_io, 'PNG')
    img_io.seek(0)

    return send_file(img_io, mimetype='image/png', as_attachment=True, download_name='processed_image.png')

@functions_framework.http
def index(request):
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
