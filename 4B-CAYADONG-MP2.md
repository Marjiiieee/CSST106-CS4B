# **Machine Problem 2 - Applying Image Processing Techniques**

![MP 2](https://github.com/user-attachments/assets/be132bbc-7d6d-44f7-b121-c6b39caf8254)

## **Perception and Computer Vision**

***Perception***

* The ability to see, hear, or become aware of something through the senses.

* Process of acquiring, interpreting, selecting, and organizing sensory information.

***Computer Vision***

* A field of computer science that focuses on enabling computers to identify and understand objects and people in images and videos.

* A field that enables machines to "see" and interpret the visual world.

***Digital Image Processing***

* Digital Image Processing focuses on two major tasks
  
  * Improvement of pictorial information for human interpretation

  * Processing of image data for storage, transmission and representation for autonomous machine perception
 
## **1. Install OpenCV**

```python
!pip install opencv-python-headless
```

## **2. Import Libraries and Image Preperation**

  A library is a collection of functions that can be added to your Python code and called as necessary, just like any other function. There is no reason to rewrite code that will perform a standard task. With libraries, you can import pre-existing functions and efficiently expand the functionality of your code.

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

def display_image (img, title="Image"):
  plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
  plt.title(title)
  plt.axis('off')
  plt.show()

def display_images (img1, img2, title1="Image 1", title2="Image2 "):
  plt.subplot(1,2,1)
  plt.imshow(cv2.cvtColor(img1,cv2.COLOR_BGR2RGB))
  plt.title(title1)
  plt.axis('off')

  plt.subplot(1,2,2)
  plt.imshow(cv2.cvtColor(img2,cv2.COLOR_BGR2RGB))
  plt.title(title2)
  plt.axis('off')

  plt.show()
```

## **3. Load Image**

```python
from google.colab import files
from io import BytesIO
from PIL import Image

uploaded = files.upload()

image_path = next(iter(uploaded))
image = Image.open(BytesIO(uploaded[image_path]))
image = cv2.cvtColor(np.array(image),cv2.COLOR_BGR2RGB)

display_image(image, "Original Image")
```

**Result**

![image](https://github.com/user-attachments/assets/df80fe49-5400-48e4-8824-da59cdd5ccc1)

## **Image Processing Technique: Scaling and Rotation**

**Image Rotation**
  Rotation is most commonly used to improve the visual appearance of an image, although it can be useful as a preprocessor in applications where directional operators are involved.

```python
def scale_image(image, scale_factor):
  height, width = image.shape[:2]
  scale_image = cv2.resize(image, (int(width * scale_factor), int(height * scale_factor)), interpolation=cv2.INTER_LINEAR)
  return scale_image

scaled_image = scale_image(image, 0.5)
display_image(scaled_image, "Scaled Image (50%)")
```

***Result***

![image](https://github.com/user-attachments/assets/d9a805c5-3f52-49cd-924b-5c9d3e6d8838)

**Image Scaling**

  Photos can make or break a website. They can either bolster credibility and legitimacy or solidify amatuer status and so it's super important to get those photos just right.

```python
def rotate_image(image, angle):
  height, width = image.shape[:2]
  center = (width // 2, height //2)
  matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
  rotated_image = cv2.warpAffine (image, matrix, (width, height))
  return rotated_image

rotated_image = rotate_image(image, 45)
display_image(rotated_image, "Rotated Image (45%)")
```

***Result***

![image](https://github.com/user-attachments/assets/bae76da5-e244-42e5-8ed4-018b57501509)

## **Image Processing Technique: Blurring Techniques**

**Motion Blur**

  In computer graphics, the simulation of motion blur is useful both in animated sequences where the blurring tends to remove temporal aliasing effects and in static images where it por- trays the illusion of speed or movement among the objects in the scene.

```python
def motion_blur(img):
  kernel_size = 15
  kernel = np.zeros((kernel_size, kernel_size))
  kernel[int((kernel_size - 1) / 2), :] = np.ones(kernel_size)
  kernel = kernel / kernel_size

  motion_blurred = cv2.filter2D(img, -1, kernel)
  return motion_blurred

motion_blurred = motion_blur(image)
plt.imshow(cv2.cvtColor(motion_blurred, cv2.COLOR_BGR2RGB))
plt.title("Motion Blur")
plt.axis('off')
plt.show()
```

***Result***

![image](https://github.com/user-attachments/assets/8ef08d8b-de89-4fb1-9d39-4a0339d021ad)


**Unsharp Mask**

  Unsharp Masking method accentuates the contrast at the edges of the core elements within an image, making those parts appear more defined and 'sharper. ' More specifically, it brings out more detail by subtracting a smoothed (blurry) version of an image from the original.

```python
#UNSHARP MASK
def unsharp_mask(img):
  blurred = cv2.GaussianBlur(img, (9, 9), 10.0)
  sharpened = cv2.addWeighted(img, 1.5, blurred, -0.5, 0)
  return sharpened

sharpened_image = unsharp_mask(image)
plt.imshow(cv2.cvtColor(sharpened_image, cv2.COLOR_BGR2RGB))
plt.title("Unsharp Mask (Sharpening)")
plt.axis('off')
plt.show()
```

***Result***

![image](https://github.com/user-attachments/assets/338ce5d9-fcf5-44a3-8161-ea0376875230)

**Gaussian Blur**

  Photographers and designers choose Gaussian functions for several purposes. If you take a photo in low light, and the resulting image has a lot of noise, Gaussian blur can mute that noise. If you want to lay text over an image, a Gaussian blur can soften the image so the text stands out more clearly.

```python
gaussian_blur = cv2.GaussianBlur(image, (15,15),0)
display_image(gaussian_blur, "Gaussian Blur")
```

***Result***

![image](https://github.com/user-attachments/assets/82d6b1e4-f0f5-4601-bb8c-6391f71ae09e)

**Median Blur**

  Median filtering is used as a smoothing technique, which is effective at removing noise in smooth patches or smooth regions of a signal. Unlike low-pass FIR filters, the median filter tends to preserve the edges in an image. Because of this, median filtering is very widely used in digital image processing.

```python
median_blur = cv2.medianBlur(image, 15)
display_image(median_blur, "Median Blur")
```

***Result***

![image](https://github.com/user-attachments/assets/7e855b9a-dda4-4e4f-aa85-d1e36f76cc7d)

**Bilateral Filter**

  The bilateral filter converts any input image to a smoothed version. It removes most texture, noise, and fine details, but preserves large sharp edges without blurring.

```python
bilateral_filter = cv2.bilateralFilter(image, 30, 60, 90)
display_image(bilateral_filter, "Bilateral Filter")
```

***Result***

![image](https://github.com/user-attachments/assets/3fccc94c-5470-4689-abdf-0bb8e215b964)

**Box Filter**

  Box Filter is a low-pass filter that smooths the image by making each output pixel the average of the surrounding ones, removing details, noise and and edges from images.

```python
box = cv2.boxFilter(image, -1, (5, 5))
display_image(box, "Box Filter")
```

***Result***

![image](https://github.com/user-attachments/assets/1b6167c9-2ea3-456a-923f-dd73857a45ad)

## **Image Processing Technique: Edge Detection Techniques**

**Canny Edge Detection**

  The Canny edge detection algorithm is a widely used and powerful technique for identifying edges in images. Its multistage process ensures accurate edge localization, low error rates, and robustness to noise.

```python
canny_edges = cv2.Canny(image, 180, 180)
display_image(canny_edges, "Canny Edge Detection")
```

***Result***

![image](https://github.com/user-attachments/assets/5f8c0749-970c-47f5-9264-9aaa815e1515)

**Sobel Edge Detection**

  The Sobel operator performs a 2-D spatial gradient measurement on an image and so emphasizes regions of high spatial frequency that correspond to edges. Typically it is used to find the approximate absolute gradient magnitude at each point in an input grayscale image.

```python
def sobel_edge_detection(img):
  gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
  sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
  sobel_combined = cv2.magnitude(sobelx, sobely)
  return sobel_combined

sobel_edges = sobel_edge_detection(image)
plt.imshow(sobel_edges, cmap='gray')
plt.title("Sobel Edge Detection")
plt.axis('off')
plt.show()
```

***Result***

![image](https://github.com/user-attachments/assets/d2d20c99-8a3d-4b80-85b8-1b47f2892e52)

**Laplacian Edge Detection**

  The Laplacian is a 2-D isotropic measure of the 2nd spatial derivative of an image. The Laplacian of an image highlights regions of rapid intensity change and is therefore often used for edge detection (see zero crossing edge detectors).

```python
def laplacian_edge_detection(img):
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  laplacian = cv2.Laplacian(gray, cv2.CV_64F)
  return laplacian

laplacian_edges = laplacian_edge_detection(image)
plt.imshow(laplacian_edges, cmap='gray')
plt.title("Laplacian Edge Detection")
plt.axis('off')
plt.show()
```

***Result***

![image](https://github.com/user-attachments/assets/3d118661-9a94-48d1-ae3c-148bec3a92c9)

**Prewitt Edge Detection**

  Prewitt edge detection is a technique used for detecting edges in digital images. It works by computing the gradient magnitude of the image intensity using convolution with Prewitt kernels. The gradients are then used to identify significant changes in intensity, which typically correspond to edges.

```python
def prewitt_edge_detection(img):
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  kernelx = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]], dtype=int)
  kernely = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]], dtype=int)
  prewittx = cv2.filter2D(gray, cv2.CV_64F, kernelx)
  prewitty = cv2.filter2D(gray, cv2.CV_64F, kernely)
  prewitt_combined = cv2.magnitude(prewittx, prewitty)
  return prewitt_combined

prewitt_edges = prewitt_edge_detection(image)
plt.imshow(prewitt_edges, cmap='gray')
plt.title("Prewitt Edge Detection")
plt.axis('off')
plt.show()
```

***Result***

![image](https://github.com/user-attachments/assets/21b9c105-82ab-4c98-b378-6521510be501)

### **What is the Importance of Image Processing?**

  The implementation of image processing techniques has had a massive impact on many tech organizations. Here are some of the most useful benefits of image processing, regardless of the field of operation:

* The digital image can be made available in any desired format (improved image, X-Ray, photo negative, etc).

* It helps to improve images for human interpretation.

* Information can be processed and extracted from images for machine interpretation.

* The pixels in the image can be manipulated to any desired density and contrast.

* Images can be stored and retrieved easily.

* It allows for easy electronic transmission of images to third-party providers.

  Image processing is done to enhance an existing image or to sift out important information from it. This is important in several Deep Learning-based Computer Vision applications, where such preprocessing can dramatically boost the performance of a model. Image processing is a versatile field with numerous applications that range from improving image quality to enabling advanced automation and analysis in various domains. Its benefits include enhancing image quality, automating tasks, and extracting valuable information for decision-making and research.

# ***Link to the Google Colab Project***

* [4B-CAYADONG-MP2.ipynb](https://colab.research.google.com/drive/1RWLROSIZxgY2uJIKZVriILw5UeauKm7D?usp=sharing)
