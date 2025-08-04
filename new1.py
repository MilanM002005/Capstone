import cv2
from ultralytics import YOLO

FOCAL_LENGTH = 615
KNOWN_WIDTH = 50

model = YOLO('yolov8n.pt')

cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("Failed to open webcam.")
    exit()

print("Webcam connected..")

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

zone_width, zone_height = 450, 300
zone_x1 = (frame_width - zone_width) // 2
zone_y1 = (frame_height - zone_height) // 2
zone_x2 = zone_x1 + zone_width
zone_y2 = zone_y1 + zone_height

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    results = model(frame, stream=True)

    object_in_zone = False  

    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            width_in_pixels = x2 - x1

            distance = (KNOWN_WIDTH * FOCAL_LENGTH) / (width_in_pixels + 1e-6)

            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            if zone_x1 < cx < zone_x2 and zone_y1 < cy < zone_y2:
                object_in_zone = True
                cv2.putText(frame, "Object Entered....", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

           
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{model.names[cls_id]} {conf:.2f}, {distance:.1f} cm"
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

   
    cv2.rectangle(frame, (zone_x1, zone_y1), (zone_x2, zone_y2), (255, 0, 0), 2)
    cv2.putText(frame, " Zone", (zone_x1, zone_y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    cv2.imshow("Zone Detection + YOLO", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
