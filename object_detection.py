import cv2
from ultralytics import YOLO


model = YOLO("yolov8n.pt")


cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Unable to access webcam.")
    exit()

print("Press 'q' to quit.")


while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Detect objects
    results = model.track(frame, persist=True)

    # Draw bounding boxes and labels
    annotated_frame = results[0].plot()

    # Display the output
    cv2.imshow("Object Detection and Tracking", annotated_frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
    
cap.release()
cv2.destroyAllWindows()