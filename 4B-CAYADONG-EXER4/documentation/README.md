# **Object Detection and Recognition**

Object detection is an important computer vision task used to detect instances of visual objects of certain classes (for example, humans, animals, cars, or buildings) in digital images such as photos or video frames. The goal of object detection is to develop computational models that provide the most fundamental information needed by computer vision applications

## **What is it used for?**

**1. Person Detection**

- Person detection is a variant of object detection used to detect a primary class “person” in images or video frames. Detecting people in video streams is an important task in modern video surveillance systems. The recent deep learning algorithms provide robust person detection results. Most modern person detector techniques are trained on frontal and asymmetric views.

**2. Transportation**

- In self-driving cars, object detection is used to identify roads, pedestrians, and traffic lights. It can also be used to identify obstacles on train tracks or to check for people in restricted areas.

**3. Contactless checkout**

- Object detection can be used to create a "virtual" shopping cart for shoppers as they pick items off the shelf. Shoppers can then pay for their items at a contactless kiosk or walk out the door.

## **Histogram of oriented gradients (HoG)**

In the case of high dimensionality, feature descriptors are used to avoid unnecessary computations involved in classification. Histogram of oriented gradients (HoG) is a feature descriptor used to define an image by the pixel intensities and intensities of gradients of pixels. Gradients define the edges of an image, so extraction of the HoG feature descriptor is the same as extracting edges.

Histogram of Oriented Gradients generates gradients at each point of the image providing invariance to occlusions, illumination, and expression changes. Group sparse coding with HoG feature descriptors is used to achieve good results on face recognition.

![HOG](https://github.com/user-attachments/assets/6bf09c17-901a-4943-9b3c-1302a8cbcbf4)

## **YOLO (You Only Look Once) Object Detection**

Object detection is a computer vision technique for identifying and localizing objects within an image or a video. Image localization is the process of identifying the correct location of one or multiple objects using bounding boxes, which correspond to rectangular shapes around the objects. This process is sometimes confused with image classification or image recognition, which aims to predict the class of an image or an object within an image into one of the categories or classes. 

![YOLO](https://github.com/user-attachments/assets/2efa2a8a-14c4-4d35-a811-c532875214a7)

## **SSD (Single Shot MultiBox Detector) with TensorFlow**

SSD is an unified framework for object detection with a single network. SSD is a deep learning-based method that detects objects in images in a single pass through a convolutional neural network (CNN). SSD is designed to perform object detection in real-time. It's faster and more efficient than other methods because it doesn't use a two-stage approach. 

![SSD](https://github.com/user-attachments/assets/a7e9bdbd-6a16-4daa-9865-353cddb27c99)

### **Why Object Detection Is Important**

Object detection, a key technology used in advanced driver assistance systems (ADAS), enables cars to detect driving lanes and pedestrians to improve road safety. Object detection is also an essential component in applications such as visual inspection, robotics, medical imaging, video surveillance, and content-based image retrieval.

### **How Object Detection Works**

You can use a variety of techniques to perform object detection. Popular deep learning–based approaches using convolutional neural networks (CNNs), such as YOLO, SSD, or R-CNN, automatically learn to detect objects within images.

*Links used in this documentation:*

* https://viso.ai/deep-learning/object-detection/#:~:text=While%20similar%2C%20object%20detection%20and,does%20not%20provide%20localization%20information.
* https://www.sciencedirect.com/topics/computer-science/histogram-of-oriented-gradient#:~:text=Histogram%20of%20oriented%20gradients%20(HoG)%20is%20a%20feature%20descriptor%20used,the%20same%20as%20extracting%20edges.
* https://www.datacamp.com/blog/yolo-object-detection-explained
* https://www.mathworks.com/discovery/object-detection.html#:~:text=Object%20detection%20is%20a%20computer,learning%20to%20produce%20meaningful%20results.
