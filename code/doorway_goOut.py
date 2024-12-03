import math
import time
import cv2
from djitellopy import Tello
from ultralytics import YOLO


def initializeTello():
    myDrone = Tello()
    myDrone.connect()
    myDrone.for_back_velocity = 0
    myDrone.left_right_velocity = 0
    myDrone.up_down_velocity = 0
    myDrone.yaw_velocity = 0
    myDrone.speed = 0
    print(myDrone.get_battery())
    myDrone.streamoff()
    myDrone.streamon()
    return myDrone


def telloGetFrame(myDrone, w=1280, h=720):
    img = myDrone.get_frame_read().frame
    img = cv2.resize(img, (w, h))
    img =  cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img

####################################################
width = 1280  # WIDTH OF THE IMAGE
height = 720  # HEIGHT OF THE IMAGE
startCounter = 0# for no Flight 1 - for flight 0
door_center_y = 0
door_center_x = 0
door_detection = False
is_rotat = 1
####################################################

myDrone = initializeTello()

# Load the YOLO model
door_model = YOLO("C:\\Users\\nayef\\Desktop\\x-project\\ptXpt.pt")

while True:
    # Flight
    if startCounter == 0:
        myDrone.takeoff()
        myDrone.send_rc_control(0, 0, 10, 0)
        startCounter = 1

    # Step 1
    img = telloGetFrame(myDrone, width, height)
    # Step 2: Detect doors using YOLO
    results = door_model(img,show =True)
    if is_rotat:
        myDrone.send_rc_control(0, 0, 0, 10)  # סיבוב ימינה במהירות 10
        time.sleep(0.1)
    if door_detection:
        is_rotat = 0

    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Bounding Box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1
            # Class Name
            cls = int(box.cls[0])
            # Check if confidence is greater than 0.45
            if box.conf[0] > 0.7 and cls== 1:
                door_detection = True
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                #cvzone.cornerRect(img, (x1, y1, w, h))
                # Confidence
                conf = math.ceil((box.conf[0] * 100)) / 100

                # Add class and confidence information inside the bounding box
                label_text = f"Class: {cls}, Confidence: {conf:.2f}"
                cv2.putText(img, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                # Calculate center of the bounding box
                door_center_x = (x1 + x2) // 2
                door_center_y = (y1 + y2) // 2
                # המרת הגודל של הדלת לסטייה מהגודל הרצוי
                size_img = width * height
                size_door = w * h

                size_error = (size_door / size_img)
                print("size error", size_error)

                # חישוב התקנה מהמרכז לדלת
                horizontal_error = door_center_x - (width// 2)
                vittcal_error = door_center_y - (height // 2)

                # כיוונים לתנועה של הדרון
                left_right = 0
                up_down = 0
                forward_backward = 0
                yaw = 0  # זווית סיבוב

                if vittcal_error > 20:
                    up_down = -5
                elif vittcal_error < -20:
                    up_down = 5

                # סבב את הדרון כך שנקודת המרכז של הדלת תהיה במרכז התמונה
                if horizontal_error > 15:
                    yaw = 5  # סיבוב שמאלה
                elif horizontal_error < -15:
                    yaw = -5  # סיבוב ימינה


                # קביעת התנועה לפי הסטייה מהגודל הרצוי
                if size_error > 0.9:
                    forward_backward = -10  # תזוזה אחורה
                elif size_error <0.8:
                        forward_backward = 10 # תזוזה קדימה

                if size_error < 0.9 and size_error > 0.8:
                    if cls == 1:
                        myDrone.send_rc_control(0, 20, 0, 0)
                        time.sleep(4)
                        myDrone.send_rc_control(0, 0, 0, 0)

                    myDrone.land()
                    print (f"finsh")
                    break

                # שליחת פקודות לדרון
                myDrone.send_rc_control(left_right, forward_backward, up_down, yaw)

    cv2.imshow('Image', img)
    time.sleep(0.05)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        myDrone.land()
        break
