# Robot Guide for the Visually Impaired

## Project Overview
This project was developed by **Naif Taha** as part of the course **'Data Learning in Real-Time Systems'**, under the supervision of **Prof. Dan Feldman** at the **University of Haifa**.

## Table of Contents
- [Introduction](#introduction)
- [Door Detection Algorithm](#door-detection-algorithm)
- [Model Usage](#model-usage)
- [Code Implementation](#code-implementation)
- [Project Results](#project-results)
- [Conclusion](#conclusion)

## Introduction
### What is a Robot Guide?
A **robot guide** is designed to assist individuals with visual impairments in navigating safely and independently. The robot relies on various sensors, such as cameras, radars, and motion detectors, to scan its surroundings and detect obstacles.

However, there are also guide robots that operate solely using a **camera**, without additional sensors. These robots scan the environment through a camera to detect **open doors**, enabling them to navigate spaces more efficiently.

### Advantages of a Camera-Only Robot Guide
Using only a camera instead of multiple sensors provides several advantages:
- **Lower Cost**: The absence of additional sensors makes production significantly cheaper.
- **Easier Operation**: Requires less maintenance compared to robots equipped with multiple sensors.
- **Portability**: The design is smaller and more lightweight, making it easy to carry.

### Challenges of a Camera-Only Robot Guide
While cost-effective and simpler, the lack of additional sensors presents challenges:
- Limited environmental awareness compared to multi-sensor robots.
- Dependency on optimal lighting conditions and clear visibility.

## Project Goal
This project aims to develop a robot that can navigate **out of a room independently** using only a camera. To achieve this, we need to:
- **Develop an algorithm** capable of detecting open doors in real-time.
- **Guide the robot** based on the door's position in the captured image.

## Open Door Detection Algorithm

### Input & Output
- **Input:** Image
- **Output:** If an open door is detected, return details such as the door's center, width, and height.

### Choosing a Model
To detect open doors, we utilize a supervised learning model, which requires labeled training data. Among various architectures like COCO, YOLO, and OpenAI models, we chose **YOLOv5X** from **Ultralytics** for this project.

YOLOv5X leverages five convolutional layers to achieve high accuracy in detecting small objects. It also incorporates new anchor mechanisms that enhance detection across varying object sizes.

For more details about YOLOv5X, visit: [YOLOv5 GitHub Repository](https://github.com/ultralytics/yolov5)

### Model Training Process
To build our model, we follow these steps:
1. Collect a dataset of images containing open doors.
2. Label the images by marking the doors.
3. Train the model on the labeled dataset.

### Dataset Preparation
We gathered **1,600 images** containing open, closed, and partially open doors. The dataset is divided into three subsets:
- **Train** – For model training.
- **Test** – For model evaluation.
- **Validation** – For hyperparameter tuning.

We used [Roboflow](https://app.roboflow.com/) to manage and preprocess the dataset. A larger dataset generally improves model accuracy.

### Annotation Process
Since we are using supervised learning, each image needs labeled bounding boxes indicating doors. Labels are assigned descriptive names such as `open` for open doors.

### Training the Model
While Roboflow provides a user-friendly training interface, it does not support exporting offline weight files (`.pt`). To solve this, we trained the model on a local machine instead.

**Challenges:**
- **Training Time:** Training locally can take over 24 hours, depending on dataset size.
- **Alternative Solution:** Training was performed using **Google Colab**, leveraging Ultralytics' training notebook. This significantly reduced training time to approximately **12 hours for 100 epochs**.

Google Colab Training Notebook: [YOLOv5 Custom Training](https://colab.research.google.com/github/roboflow-ai/yolov5-custom-training-tutorial/blob/main/yolov5-custom-training.ipynb)

### Training Execution
Ultralytics provides direct integration between Google Colab and Roboflow, allowing seamless dataset transfer and annotation. Another option is to manually upload the dataset directly to Ultralytics' training environment.

The key training parameter to optimize is **epochs**—the number of times the model iterates over the dataset. Setting an appropriate epoch count is crucial to prevent overfitting. Based on our experience, an optimal range is **100-150 epochs**, depending on dataset size.

### Training Outcomes
At the end of training, we obtain a **weights file (`something.pt`)**, which allows the model to function offline.

#### Key Training Issues & Solutions:
- **False Positives:** The first model incorrectly detected windows as doors.
- **Labeling Issues:** The second model detected doors but misidentified the exact open area, leading to navigation errors.
- **Transparent Doors:** The final model struggled with glass doors, which were resolved by adding transparent door images to the dataset.

---

In the next stage, we integrate the trained model into our project and address real-time processing challenges.


## Model Usage
Here’s your entire text translated into English and formatted as a single `.md` file, ready to be copied directly to GitHub:


## Introduction
To use the model, we will use the Ultralytics library.

### Installing the Library:
```bash
pip install ultralytics
```

### Loading the Model and Running Detection on an Image:
```python
from ultralytics import YOLO

model = YOLO("C:\\Users\\nayef\\Desktop\\x-project\\ptXpt.pt")
results = model(img, stream=True)
```

This is how the model is used. When a door is detected, data about its location in the image can be obtained, such as the two parallel endpoints. Using these points, the width, height, and center point of the door can be calculated.

## Model Results
The model detects an open door with high accuracy, but there is an issue with real-time data processing. Since the model is powerful and detects small objects, it requires high processing power, which causes significant slowdowns.

**Link to a demonstration of the model's results on the CPU:**  
[https://screenrec.com/share/CfsxEzLZV9](https://screenrec.com/share/CfsxEzLZV9)

In the output, you can see that running a short video clip (15 seconds) takes more than a minute, indicating a performance issue.

## Real-Time Data Processing
At this stage, we will solve the issue of the model's slow performance on the CPU.

### Solution: Using a Graphics Card (GPU)
In our project, we will use an **NVIDIA** GPU.

### Setting Up the Appropriate Environment:
1. Install CUDA 11.8
2. Install cuDNN 11.x
3. Copy cuDNN files to the CUDA 11.8 directory
4. Install PyTorch compatible with CUDA 11.8

After completing these steps, the model will automatically run on the GPU if available; otherwise, it will run on the CPU.

**Link to a demonstration of the difference between running on GPU vs. CPU:**  
[https://screenrec.com/share/xp5oKXHigW](https://screenrec.com/share/xp5oKXHigW)

## Summary
At this stage, we have a self-detection model that identifies an open door in real time. In the next step, we will move to the final navigation code, where we will use a **TELLO** drone.


### How to Use:
1. Copy the entire content above.
2. Go to GitHub and create a new file (e.g., `README.md` or `documentation.md`).
3. Paste the content into the file.
4. Commit the changes.

This file is now ready for GitHub and will render properly with Markdown formatting. Let me know if you need further assistance! 😊

## Code Implementation
Here’s the translated and professionally rewritten version of your text, formatted for a `.md` file. It is concise, professional, and ready to be added to GitHub:

## Code Introduction
At this stage, we are ready to transition to the code implementation.

### Requirements:
- A computer.
- A **TELLO** drone.

### Notes:
- This code is tailored for the **TELLO** drone used in the lab. The drone's camera quality is not as high as the samples used to train the model, so we will make adjustments to improve the results. Specifically, we will resize the image to **1280x720**, which has proven to be the optimal size for this setup. However, if using a higher-quality camera (e.g., FHD or above), the recommended image size is **640x640**.
- The rationale for resizing the image is based on two key observations:
  1. **Self-detection models perform better with larger samples.**
  2. **Models process smaller images faster.**
  Given that our model is powerful and fast, we can sacrifice a small amount of speed to achieve better detection accuracy.
- Additionally, we will implement a **centering navigation algorithm** to further enhance performance.

---

## Algorithm Overview
1. **Image Capture**: Receive an image from the drone.
2. **Model Inference**: Run the model on the image to detect the door. The model returns the coordinates of the bounding box: `(x1, y1)` (top-left corner) and `(x2, y2)` (bottom-right corner).
3. **Centering Calculation**: Calculate the distance between the center of the door and the center of the image. Adjust the drone's position to align the door's center with the image's center.
4. **Size-Based Navigation**: Calculate the size of the door relative to the image. Move the drone closer to the door if the door appears small in the image.
5. **Exit Condition**: If the door is no longer detected, the drone exits the door.

---

## Code Implementation

### Libraries Used:
- **`ultralytics`**: For loading and running the YOLO model. It automatically utilizes the GPU if available.
- **`cv2` (OpenCV)**: For image processing and visualization.
- **`math`**: For calculating confidence scores.
- **`time`**: For adding delays between drone actions.
- **`djitellopy`**: For controlling the **TELLO** drone.

### Key Functions:
1. **`initializeTello()`**:
   - Initializes the drone connection.
   - Sets initial velocities and speed to zero.
   - Checks the drone's battery level.
   - Enables the drone's video stream.

2. **`telloGetFrame(myDrone, w=1280, h=720)`**:
   - Captures a frame from the drone.
   - Resizes the frame to the specified dimensions (`1280x720` by default).
   - Converts the frame from RGB to BGR format for OpenCV compatibility.

---

### Main Workflow

#### Parameters:
- `width = 1280`: Image width.
- `height = 720`: Image height.
- `startCounter = 0`: Controls the drone's takeoff (0 = no flight, 1 = flight).
- `door_center_x`, `door_center_y`: Coordinates of the door's center.
- `door_detection = False`: Tracks whether a door is detected.
- `is_rotat = 1`: Controls the drone's rotation.

#### Initialization:
```python
myDrone = initializeTello()
door_model = YOLO("C:\\Users\\nayef\\Desktop\\x-project\\ptXpt.pt")
```

#### Main Loop:
1. **Takeoff**:
   - The drone takes off and begins searching for the door.
   ```python
   if startCounter == 0:
       myDrone.takeoff()
       myDrone.send_rc_control(0, 0, 10, 0)
       startCounter = 1
   ```

2. **Image Capture and Processing**:
   - Capture a frame from the drone.
   - Run the YOLO model on the frame.
   ```python
   img = telloGetFrame(myDrone, width, height)
   results = door_model(img, show=True)
   ```

3. **Door Detection**:
   - Extract bounding box coordinates (`x1, y1, x2, y2`).
   - Calculate the door's width and height.
   - Filter detections with a confidence score above 75%.
   ```python
   for r in results:
       boxes = r.boxes
       for box in boxes:
           x1, y1, x2, y2 = box.xyxy[0]
           x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
           w, h = x2 - x1, y2 - y1
           cls = int(box.cls[0])
           if box.conf[0] > 0.75 and cls == 1:
               door_detection = True
   ```

4. **Navigation Logic**:
   - Calculate the door's center and its deviation from the image's center.
   - Adjust the drone's position based on the deviation.
   ```python
   door_center_x = (x1 + x2) // 2
   door_center_y = (y1 + y2) // 2
   size_img = width * height
   size_door = w * h
   size_error = (size_door / size_img)
   horizontal_error = door_center_x - (width // 2)
   vertical_error = door_center_y - (height // 2)
   ```

5. **Drone Movement**:
   - Move the drone forward, backward, up, down, or rotate based on the calculated errors.
   ```python
   if size_error > 0.9:
       forward_backward = -10  # Move backward
   elif size_error < 0.8:
       forward_backward = 10   # Move forward
   myDrone.send_rc_control(left_right, forward_backward, up_down, yaw)
   ```

6. **Exit Condition**:
   - If the door is no longer detected, land the drone.
   ```python
   if size_error < 0.9 and size_error > 0.8:
       if cls == 1:
           myDrone.send_rc_control(0, 20, 0, 0)
           time.sleep(4)
           myDrone.land()
           break
   ```

---

## Summary
In this stage, we implemented a navigation algorithm that centers the drone relative to the detected door. The algorithm is linear in complexity and easy to validate. While the **TELLO** drone's camera quality is suboptimal, the code is designed to work effectively with it. For higher-quality cameras, the image size can be adjusted to **640x640** for better performance.

In the next stage, we will present the practical results of the project. Notably, the model's detection accuracy is lower with the drone's camera compared to the high-quality samples used during training. This highlights the importance of camera quality in achieving consistent high-confidence detections.


## Project Results
## Project Results: Introduction
In this stage, we will present the results of the project. Previously, we demonstrated the model's performance, so this section focuses on the final outcome of the door-exit operation and key research insights.

---

## Final Results
### Door Exit Operation
The drone successfully navigates through the door using the following steps:
1. **Detection**: The model identifies the door with high confidence.
2. **Centering**: The drone aligns itself with the center of the door.
3. **Navigation**: The drone moves forward or backward based on the door's size in the image.
4. **Exit**: Once the drone passes through the door, it lands safely.

**Key Observations**:
- The model achieves a detection confidence of over **75%** in most cases.
- The navigation algorithm effectively centers the drone relative to the door.
- The drone's movement is smooth and controlled, demonstrating the algorithm's robustness.

---

## Research Insights
1. **Camera Quality Matters**:
   - The **TELLO** drone's camera quality significantly impacts detection accuracy. While the model performs well with high-quality images, the drone's lower-resolution camera results in occasional missed detections.
   - For future improvements, using a drone with a higher-quality camera (e.g., FHD or 4K) is recommended.

2. **Image Size Optimization**:
   - Resizing the image to **1280x720** improved detection accuracy for the **TELLO** drone. However, for higher-quality cameras, a smaller image size (**640x640**) is more efficient without sacrificing accuracy.

3. **Real-Time Performance**:
   - Running the model on a **GPU** significantly improved real-time performance. On a CPU, the processing time was too slow for practical use.
   - The navigation algorithm's linear complexity ensures efficient performance even with limited computational resources.

4. **Limitations**:
   - The model occasionally fails to detect the door in low-light conditions or when the door is partially obscured.
   - The drone's battery life limits the duration of continuous operation.

---

## Conclusion
The project successfully demonstrates the integration of a YOLO-based detection model with a **TELLO** drone for autonomous navigation through a door. While the results are promising, there is room for improvement, particularly in enhancing camera quality and optimizing the model for real-world conditions.

In the next phase, we will explore advanced techniques to address these limitations and further improve the system's reliability and performance.

## Project Documentation: Tasks, Challenges, and Future Research

### Project Overview
The research focuses on identifying a door in a 2D image and navigating a robot through it in a 3D space. The project involves:
1. Programming a real-time **object detection model** to identify doors.
2. Developing a navigation algorithm to center the robot relative to the door.

---

### Project Tasks
1. **Train a Robust Machine Learning Model**:
   - Develop an object detection model capable of identifying doors in real time.
2. **Offline Functionality**:
   - Ensure the model operates without an internet connection.
3. **Transparent Door Detection**:
   - Enable the model to detect transparent or glass doors.
4. **Safe Navigation Algorithm**:
   - Design an algorithm that ensures safe and environment-appropriate drone movement.
5. **Optimize for Low-Quality Camera**:
   - Address the limitations of the drone's low-quality camera.

---

### Challenges
1. Training a powerful machine learning model.
2. Ensuring the model works offline.
3. Detecting transparent doors.
4. Developing a safe and efficient navigation algorithm.
5. Overcoming the limitations of the drone's low-quality camera.

---

### Future Research Directions
During the project, I considered several extensions for future research:

#### 1. **Outdoor Navigation Robot for the Visually Impaired**
   - **Objective**: Develop a robot that can guide visually impaired individuals outdoors.
   - **Features**:
     - Detect crosswalks and traffic lights.
     - Wait for a green light before crossing.
     - Navigate to a bus stop and identify the arriving bus number.
   - **Implementation**: Add relevant models for each task (e.g., bus detection, number recognition, door detection).

#### 2. **Real-Time Doorway Detection in Low-Power Systems**
   - **Objective**: Detect doors in real time using a 2D camera on low-power devices.
   - **Approach**:
     - Use multiple models: one for detecting the presence of a door, one for depth estimation, and one for color analysis.
     - Combine these models to achieve door detection without requiring a GPU.

---

### Why I Deserve a High Grade
This is a challenging question, and the lecturer may have a better perspective. However, here are my thoughts:
1. **Innovative Project**:
   - The project has the potential to help many people and save significant costs (e.g., guide dogs cost over 20,000 ILS).
2. **Powerful and Reliable Model**:
   - The model I designed is robust and trustworthy. It is an advanced version (v16) of existing models, requiring 12 hours of training and additional testing for each iteration.
   - The model can process **30 frames per second**, making it highly efficient for real-time applications.
3. **Effort and Problem-Solving**:
   - I invested significant effort in the course and demonstrated strong problem-solving skills.

---

### Acknowledgments
- **Course Lecturer**: Prof. Dan Feldman  
  Helped with decision-making and problem-solving strategies.
- **Course TA**: Fares Fares  
  Assisted with the practical aspects of using the drone.

---

### Note
All content in the presentation, including code, models, images, claims, and any other materials in the project folder, is my original work. No external sources were used without proper attribution.

Thank you!

