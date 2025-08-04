# This script uses two webcams to detect objects in a central zone using YOLOv8.
# It combines the views from both cameras and performs object detection on the merged image.

import cv2
from ultralytics import YOLO

FOCAL_LENGTH = 615
KNOWN_WIDTH = 50

model = YOLO('yolov8n.pt')

# Open two webcams (adjust indices as needed)
cap1 = cv2.VideoCapture(1, cv2.CAP_DSHOW)  # Left-side camera
cap2 = cv2.VideoCapture(2, cv2.CAP_DSHOW)  # Right-side camera

if not cap1.isOpened() or not cap2.isOpened():
    print("Failed to open both cameras.")
    exit()

print("Both cameras connected. Press 'q' to exit.")

# Get frame size from cam1 (assuming both are same)
frame_width = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Central zone in merged view (total width = frame_width * 2)
zone_w, zone_h = 600, 350
zone_x1 = (frame_width * 2 - zone_w) // 2
zone_y1 = (frame_height - zone_h) // 2
zone_x2 = zone_x1 + zone_w
zone_y2 = zone_y1 + zone_h

while True:
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()

    if not ret1 or not ret2:
        print("Failed to grab frames.")
        break

    # Combine left and right views side-by-side
    combined_frame = cv2.hconcat([frame1, frame2])

    # Run YOLOv8 detection on combined image
    results = model(combined_frame, stream=True)

    object_in_zone = False

    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            width_in_pixels = x2 - x1

            distance = (KNOWN_WIDTH * FOCAL_LENGTH) / (width_in_pixels + 1e-6)

            # Check if object's center is inside the center zone
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            if zone_x1 < cx < zone_x2 and zone_y1 < cy < zone_y2:
                object_in_zone = True
                cv2.putText(combined_frame, "Object Entered Zone!", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

            # Draw bounding box and label
            cv2.rectangle(combined_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{model.names[cls_id]} {conf:.2f}, {distance:.1f} cm"
            cv2.putText(combined_frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

    # Draw center detection zone on combined image
    cv2.rectangle(combined_frame, (zone_x1, zone_y1), (zone_x2, zone_y2), (255, 0, 0), 2)
    cv2.putText(combined_frame, "Detection Zone", (zone_x1, zone_y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Display final combined feed
    cv2.imshow("Combined View - Zone Detection", combined_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap1.release()
cap2.release()
cv2.destroyAllWindows()
