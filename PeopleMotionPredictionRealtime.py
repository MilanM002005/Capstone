import cv2
import numpy as np
import os
from ultralytics import YOLO

# Load YOLOv8 Model
model = YOLO("yolov8x.pt") 

class KalmanFilter:
    def __init__(self):
        self.kf = cv2.KalmanFilter(4, 2) 
        self.kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        self.kf.transitionMatrix = np.array([
            [1, 0, 1, 0], 
            [0, 1, 0, 1], 
            [0, 0, 1, 0], 
            [0, 0, 0, 1]
        ], np.float32) 

    def predict(self, coordX, coordY):
        measured = np.array([[np.float32(coordX)], [np.float32(coordY)]])
        self.kf.correct(measured)
        predicted = self.kf.predict()

        x_multiplier = 3.0  # Increase this for stronger left/right prediction
        px = int(predicted[0] + x_multiplier * predicted[2])  # Exaggerate x-axis motion
        py = int(predicted[1]) 

        return px, py

# Video capture settings
dispW, dispH = 640, 480
cap = cv2.VideoCapture(1,cv2.CAP_DSHOW)  # Webcam input

kf = KalmanFilter()

output_folder = "output"
os.makedirs(output_folder, exist_ok=True)

output_path = os.path.join(output_folder, "people_motion_tracking.mp4")
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # MP4 codec
outVid = cv2.VideoWriter(output_path, fourcc, 30, (dispW, dispH))

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    if not ret:
        break

    # Detect objects using YOLO
    results = model(frame)
    
    for result in results:
        for box in result.boxes:
            cls = int(box.cls[0])  
            if cls == 0:  # Class 0 = Person
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

                # Predict future position
                px, py = kf.predict(cx, cy)

                # Maintain bounding box size at predicted location
                width, height = x2 - x1, y2 - y1
                px1, px2 = px - width // 2, px + width // 2
                py1, py2 = py - height // 2, py + height // 2  

                #  actual bounding box (Green)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)

                #  predicted bounding box (Red)
                cv2.rectangle(frame, (px1, py1), (px2, py2), (0, 0, 255), 3)

    cv2.imshow("People Motion Prediction", frame)
    outVid.write(frame)  # Save the output video

    if cv2.waitKey(1) & 0xFF == 27:  
        break

cap.release()
outVid.release()
cv2.destroyAllWindows()

print(f"Video saved at: {output_path}")
