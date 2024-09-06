# **CSST106-CS4B**

https://github.com/user-attachments/assets/8c681fc4-1e06-41cc-86c3-af77c46f9d9d

## **Computer Vision**

At its core, **computer vision** is the ability of computers to understand and analyze visual content in the same way humans do. This includes tasks such as recognizing objects and faces, reading text and understanding the context of an image or video. 

Computer vision is closely related to artificial intelligence (AI) and often uses AI techniques such as machine learning to analyze and understand visual data. Machine learning algorithms are used to “train” a computer to recognize patterns and features in visual data, such as edges, shapes and colors. 
Once trained, the computer can use this knowledge to identify and classify objects in new images and videos. The accuracy of these classifications can be improved over time through further training and exposure to more data.

The computer vision system consists of two main components: a sensory device, such as a camera, and an interpreting device, such as a computer. The sensory device captures visual data from the environment and the interpreting device processes this data to extract meaning.
  
Computer vision algorithms are based on the hypothesis that “our brains rely on patterns to decode individual objects.” Just as our brains process visual data by looking for patterns in the shapes, colors and textures of objects, computer vision algorithms process images by looking for patterns in the pixels that make up the image. These patterns can be used to identify and classify different objects in the image.

To analyze an image, a computer vision algorithm first converts the image into a set of numerical data that can be processed by the computer. This is typically done by dividing the image into a grid of small units called pixels and representing each pixel with a set of numerical values that describe its color and brightness. These values can be used to create a digital representation of the image that can be analyzed by the computer.

Image processing plays a critical role in AI, particularly in enhancing, manipulating, and analyzing visual data. This is crucial because visual data, like images and videos, often contains valuable information that needs to be extracted, processed, and interpreted for various applications.

*Key Roles of Image Processing in AI:*

![Image Enhancement](https://github.com/user-attachments/assets/7dddfe4e-2de6-4313-b8dd-0abfe1c89127)

* **Image Enhancement:** AI-driven image processing techniques can improve the quality of images by reducing noise, increasing resolution, adjusting contrast, and correcting colors. This is particularly important in fields like medical imaging, where clear and detailed images are essential for accurate diagnoses.

* **Image Analysis:** AI models, particularly those based on deep learning, can analyze images to identify patterns, recognize objects, and even interpret scenes. This capability is widely used in facial recognition, autonomous vehicles, and content moderation on social media platforms.

* **Image Manipulation:** AI can also manipulate images to create new content or alter existing ones. Techniques like Generative Adversarial Networks (GANs) allow for the generation of realistic synthetic images, which have applications in creative industries, marketing, and more.

* **Real-World Applications:** AI-enhanced image processing is used in a variety of real-world applications, from improving e-commerce product images to generating visual content for advertisements. Additionally, AI-powered image search engines enable users to find specific images quickly by analyzing the content and meaning of visuals.

## **Overview of Image Processing Techniques**

In image processing, several key techniques are essential for enhancing, manipulating, and analyzing images. Some of the most important techniques include:

* **Filtering:** This involves manipulating an image to enhance certain features or remove unwanted components. Common filtering techniques include Gaussian filtering, which smoothens an image by reducing noise, and Median filtering, which is effective in removing "salt-and-pepper" noise while preserving edges. Filtering is crucial in various fields, including medical imaging, where it improves image quality for better diagnosis.

<img width="599" alt="Image Filtering" src="https://github.com/user-attachments/assets/715d0f81-870f-43d7-bdf5-d3c3eb089bc6">

* **Edge Detection:** This technique is used to identify significant transitions in intensity within an image, often representing boundaries of objects. Methods like the Sobel, Prewitt, and Canny edge detectors are widely used. Edge detection is fundamental in applications such as object recognition and image segmentation, where defining object boundaries is critical.

![Edge Detection](https://github.com/user-attachments/assets/b74c33fe-bd0a-405b-aa8b-1f66de0e5552)

* **Segmentation:** Image segmentation divides an image into meaningful parts, typically to isolate objects or regions of interest. Techniques range from traditional methods like thresholding and clustering (e.g., K-means) to advanced methods using deep learning, such as U-Net and SegNet. Segmentation is vital in medical imaging, satellite image analysis, and automated inspection systems.

![Image Segmentation](https://github.com/user-attachments/assets/fe8d7876-dac5-4b15-9a1c-0c62ae9dc132)

## **Computer Vision and Machine Learning**

**Computer vision (CV)** is concerned with giving the computer the ability to process and analyse visual content such as 2D, videos, and 3D images. CV tasks can be broadly categorised into: image classification, object detection and recognition from images, and image segmentation tasks. Image classification tasks are considered among the most common CV problems. These are widely used, especially in the medical domain, and are often formulated as a supervised machine learning (ML) problem.

CNN-based methods significantly advanced the field of CV, particularly in the areas of medical image analysis and classification. CNNs have the ability to capture the underlying representation of the images using partially connected layers and weights sharing.

![Screenshot 2024-09-06 203124](https://github.com/user-attachments/assets/09c7f277-fa9f-42ee-b9d2-56bfcbd7e99d)

Overall, object detection models consist of localisation and identification tasks. The localisation task leads to localising the object position in the image using a bounding box or mask to define which pixels within the image depict the object of interest. The identification task refers to recognise the objects referring to specific pre-defined categories, or to classify the object within the bounding box. Object detection algorithms are commonly used in the medical image analysis domain in order to detect the initial abnormality symptoms of patients.

* Image segmentation refers to a pixel-wise classification task that segment an image into areas with the same attributes. The goal of medical image segmentation is to find the region or contour of a body organ or anatomical parts in images. 

![Screenshot 2024-09-06 203212](https://github.com/user-attachments/assets/e38f8220-cf7d-476f-9711-c3809b671463)

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load image in grayscale
image = cv2.imread('Anya Forger.jpg', 0)

# Apply Canny edge detection
edges = cv2.Canny(image, 100, 200)

# Display the original image and edges
plt.figure(figsize=(10,5))
plt.subplot(121), plt.imshow(image, cmap='gray')
plt.title('Original Image')
plt.subplot(122), plt.imshow(edges, cmap='gray')
plt.title('Edge Image')
plt.show()
```
**Edge detection** identifies boundaries within an image by highlighting areas of strong gradients. The Canny algorithm is one of the most commonly used edge detectors. Used in object detection systems, like identifying shapes in robotics and is useful in medical imaging for identifying edges of tissues or organs. Its crucial in feature extraction for tasks like shape recognition and object tracking.

```python
import cv2
import matplotlib.pyplot as plt

# Load image in grayscale
image = cv2.imread('Anya Forger.jpg', 0)

# Apply global thresholding
_, thresholded = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

# Display the results
plt.figure(figsize=(10,5))
plt.subplot(121), plt.imshow(image, cmap='gray')
plt.title('Original Image')
plt.subplot(122), plt.imshow(thresholded, cmap='gray')
plt.title('Thresholded Image')
plt.show()
```
**Thresholding** is a simple method of image segmentation that converts an image into a binary form, where pixel values are set to either 0 or 1 based on a threshold value. It is often used in medical imaging to isolate different parts of an image, like distinguishing between background and foreground regions (e.g., tumor detection) and frequently applied in OCR (Optical Character Recognition) systems to process text documents. It aids in classifying regions of interest (e.g., background vs. foreground).

```python
import cv2
import matplotlib.pyplot as plt

# Load the image
image = cv2.imread('Anya Forger.jpg')

# Apply Gaussian blur
blurred = cv2.GaussianBlur(image, (15, 15), 0)

# Display original and blurred images
plt.figure(figsize=(10,5))
plt.subplot(121), plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title('Original Image')
plt.subplot(122), plt.imshow(cv2.cvtColor(blurred, cv2.COLOR_BGR2RGB))
plt.title('Blurred Image')
plt.show()
```
**Blurring** an image smoothens out sharp transitions in intensity. The Gaussian blur technique is widely used to reduce noise or prepare an image for further processing, such as edge detection. Preprocessing step in face recognition systems to reduce noise, this also helps detect large-scale features by smoothing out finer details. It reduces noise, making it easier for AI systems to detect meaningful patterns in an image.
