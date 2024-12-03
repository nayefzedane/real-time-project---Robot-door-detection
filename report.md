# Algorithm for detecting an open door

**Introduction:**  
**Input:** An image.  
**Output:** If there is an open door in the image, return information about the door, such as its center, width, and height.  

For this purpose, we will use a self-detection model. According to the terminology of machine learning, there are two types of models we can build: supervised learning and unsupervised learning. In this project, we will use supervised learning because this type requires a smaller dataset for training.  

There are many architectures for training models, such as COCO, YOLO, OPEN AI, and more.  
In this project, we will use the YOLOv5X computer vision model by Ultralytics.  

YOLOv5X uses 5 layers. This allows it to detect smaller objects with higher accuracy. Additionally, YOLOv5X uses new anchors, which better match objects of various sizes.  

For more information about the model: [https://github.com/ultralytics/yolov5](https://github.com/ultralytics/yolov5).  

To build the model, the following steps are required:
1. Collect a dataset of images containing open doors.  
2. Annotate the samples in the images (mark the open doors).  
3. Train the model on the dataset to obtain the final model.  

**Dataset**  
At this stage, I have collected 1,600 images containing open, closed, and partially open doors.  
The dataset for the training phase will be divided into three categories:  
1. Train  
2. Test  
3. Validate  

To work with this dataset, we will use the Roboflow platform to store all the images in one place.  

*Note:* A larger dataset results in a more accurate model.  
**Recommendation:** Using supervised learning requires a smaller dataset compared to unsupervised learning, depending on the algorithm being used.  

**Roboflow website:** [https://app.roboflow.com/door-open/findforway/upload](https://app.roboflow.com/door-open/findforway/upload)    
![img.png](img.png)


**Labeling Samples from the Images**  
At this stage, each image is reviewed, and the doors are labeled. This is how supervised learning works. In unsupervised learning, this step is not performed.  
When choosing a label, it is necessary to assign a name to the sample, for example, "open".  

![img_1.png](img_1.png)  


**Model Training**  
At this stage, we begin building the model.  
The Roboflow platform allows good training but does not provide a weight file, which is a problem.  

**Weight file:** A file that works without an internet connection (offline mode).  

The solution to this problem is to train the dataset on a local computer.  
Training data on a computer takes a lot of time, and it can take more than 24 hours depending on the size of the dataset.  

An alternative solution is to train the data on Google Colab using a page created by Ultralytics.  
**Advantages of using Google Colab:** Time-saving. It takes approximately 12 hours for 100 epochs, although the time depends on the size of the dataset and the number of training steps.  

**Training page:** [https://colab.research.google.com/github/roboflow-ai/yolov5-custom-training-tutorial/blob/main/yolov5-custom-training.ipynb](https://colab.research.google.com/github/roboflow-ai/yolov5-custom-training-tutorial/blob/main/yolov5-custom-training.ipynb)  

![img_2.png](img_2.png)  

How the training is done:  
Ultralytics, in their Google Colab page in collaboration with Roboflow, allows transferring the dataset with annotations for training. Another option is to upload the dataset directly to this page and perform the annotation step without relying on Roboflow. Additionally, you can use Ultralytics' website.  

What matters to us at this stage is the epochs, which is the number of times the dataset is iterated over during each training step. This is important because the appropriate number of epochs needs to be chosen to avoid overfitting, where the model recognizes only the samples it was trained on, and to ensure the model can identify new samples. From my experience, the epochs should be between 100 and 150, depending on the dataset size. Other aspects of the training are not relevant for this project and can be left at their default values.  
![img_3.png](img_3.png)  

Training takes a lot of time, and it depends on the size of the dataset.  
![img_4.png](img_4.png)  

Training results:  
The images display the training results and show that the object detection accuracy has increased. Our goal is for the detection accuracy to reach over 80%.  
![img_5.png](img_5.png)  

![img_6.png](img_6.png)  

Summary:  
At the end, we obtain a weight file called `something.pt`, which works in offline mode.  
Now, we can use the model in our project.  

In the next stage, we will use the model in the code and present a major issue regarding real-time performance when using a powerful model.  

**Important:** Training is not always successful on the first attempt. For example, in this project, I tried multiple times before achieving my final model.  

During the training process, several issues arose, which I resolved, such as:  
1. The first model identified every window as a door.  
2. The second model identified every door, but the problem was that the center point of the label was not at the door's opening, which could cause navigation issues.  
3. A more advanced model had difficulty detecting transparent doors, such as the lab door. The solution was to add transparent doors to the dataset.    

# Using the model

Introduction:  
To use the model, we will utilize the `ultralytics` library.  

**Library installation:**  
`pip install ultralytics`  

```python
from ultralytics import YOLO
model = YOLO("file path /file name.pt")
results = model(img, stream=True)
```

This is how the model is used. When a door is detected, we can extract information about its location in the image, such as the two bounding points. Using these two points, we can calculate the width, height, and center point of the door.    

![img_7.png](img_7.png)  

Model result:  
The model detects an open door with high accuracy, but there is an issue with obtaining data in real time. Since the model is powerful and capable of detecting small objects, it requires high-performance mode to operate. In the next stage, we will address this issue.  

This video demonstrates the model running on the processor.  

*Note:* The processor is an Intel i9 13th generation, considered a powerful processor, and this will showcase how strong this model is.  

**Link to the model's output on the processor:**  
[https://screenrec.com/share/CfsxEzLZV9](https://screenrec.com/share/CfsxEzLZV9)  

In the code's output, there is a short 15-second video. Running the model on just half the video's duration took more than a minute, which is a significant problem. The solution will be addressed in the next stage.    
