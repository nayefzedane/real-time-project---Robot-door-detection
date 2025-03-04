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
*(A breakdown of the code and key functions implemented.)*

## Project Results
*(Presentation of findings, images, and performance metrics.)*

## Conclusion
A camera-only guide robot presents a cost-effective and accessible solution for visually impaired individuals. Despite its limitations, it provides a viable way to assist in navigation. This project successfully demonstrates an approach to **detect open doors in real-time and guide the robot toward the exit**.


**Note**: Images will replace video demonstrations where applicable.

