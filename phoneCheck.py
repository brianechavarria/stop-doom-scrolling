import cv2
from ultralytics import YOLO
import numpy as np
import fiftyone as fo
import fiftyone.zoo as foz

# Open the default camera
cam = cv2.VideoCapture(0)

# Get the default frame width and height
frame_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (frame_width, frame_height))

# Load object detection model
model = YOLO("yolo26n.pt")


while True:
    ret, frame = cam.read()

    # Write the frame to the output file
    out.write(frame)

    # Detect objects in the frame
    results = model(frame)
    # Parse the results
    boxes = results[0].boxes.xyxy.cpu().numpy()
    confs = results[0].boxes.conf.cpu().numpy()
    labels = results[0].boxes.cls.cpu().numpy().astype(int)

    # Draw bounding boxes and labels on the image
    for box, conf, label in zip(boxes, confs, labels):
        if conf >= 0.2:  # Confidence threshold
            x1, y1, x2, y2 = map(int, box)
            label_name = model.names[label]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f'{label_name} {conf:.2f}', (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
    

    # Display the captured frame
    cv2.imshow('Camera', frame)

    # Press 'q' to exit the loop
    if cv2.waitKey(1) == ord('q'):
        break

# Release the capture and writer objects
cam.release()
out.release()
cv2.destroyAllWindows()