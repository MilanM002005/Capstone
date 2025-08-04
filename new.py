import cv2
from ultralytics import YOLO

FOCAL_LENGTH = 615
KNOWN_WIDTH = 50    

model = YOLO('yolov8n.pt')

cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)

if not cap.isOpened():
    print(" Failed to open webcam.")
    exit()

print("Webcam connected..")

while True:
    ret, frame = cap.read()
    if not ret:
        print(" Failed to grab.")
        break

    
    results = model(frame, stream=True)

    
    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            width_in_pixels = x2 - x1

           
            distance = (KNOWN_WIDTH * FOCAL_LENGTH) / (width_in_pixels + 1e-6) 

            
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{model.names[cls_id]} {conf:.2f}, {distance:.1f} cm"
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

    
    cv2.imshow("YOLO Detection + Distance", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
