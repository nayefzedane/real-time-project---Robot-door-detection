from ultralytics import YOLO
import cv2
import cvzone
import math
import time
import torch

# cap = cv2.VideoCapture(1)  # For Webcam
# cap.set(3, 1280)
# cap.set(4, 720)
cap = cv2.VideoCapture("C:\\Users\\nayef\\Downloads\\WhatsApp Video 2023-09-03 at 19.08.48.mp4")  # For Video


model = YOLO("C:\\Users\\nayef\\Desktop\\x-project\\ptXpt.pt")

classNames = ["closed","open","simi"
              ]

prev_frame_time = 0
new_frame_time = 0
while True:
    new_frame_time = time.time()
    success, img = cap.read()
    img = "C:\\Users\\nayef\\Downloads\\20230724_110911.jpg"
    #img = cv2.imread(img)
    # Resize the image to a fixed size of 640x640
    img = cv2.resize(img, (640, 640))
    results = model(img, stream=True)
    time.sleep(0.1)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Bounding Box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1
            # Check if confidence is greater than 0.45
            if box.conf[0] > 0.7:
                cvzone.cornerRect(img, (x1, y1, w, h))
                # Confidence
                conf = math.ceil((box.conf[0] * 100)) / 100
                # Class Name
                cls = int(box.cls[0])
                cvzone.putTextRect(img, f'{classNames[cls]} {conf}', (max(0, x1), max(35, y1)), scale=1, thickness=1)
                # Calculate center of the bounding box
                box_center_x = (x1 + x2) // 2
                box_center_y = (y1 + y2) // 2

                # Draw a circle at the box center
                cv2.circle(img, (box_center_x, box_center_y), 5, (0, 255, 0), -1)  # Green circle

                # Draw a circle at point (0, 0)
                cv2.circle(img, (599, 599), 5, (255, 0, 0), -1)  # Blue circle

    fps = 1 / (new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time
    print(fps)

    cv2.imshow("Image", img)
    cv2.waitKey(1)

