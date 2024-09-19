# **Exercise 1: Image Processing Techniques**

![Intro](https://github.com/user-attachments/assets/4b7600a6-1353-4474-8dde-16f609b43639)

**Comparison**
* [4B-CAYADONG-EXER1.pdf](https://github.com/user-attachments/files/17062910/4B-CAYADONG-EXER1.pdf)

## **1. Install OpenCV**

```python
!pip install opencv-python-headless
```

OpenCV is a powerful, open-source library designed specifically for image processing, offering tools like filtering, edge detection, and feature extraction. It is optimized for performance and works across various platforms, making it suitable for real-time applications. Its integration with other libraries and strong community support make it a go-to choice for both simple and advanced image processing tasks.

## **2. Import Libraries**

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

def display_image(img,title="Image"):
  plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
  plt.title(title)
  plt.axis('off')
  plt.show()

def display_images(img1,img2, title1="Image 1",title2="Image 2"):
  plt.subplot(1,2,1)
  plt.imshow(cv2.cvtColor(img1,cv2.COLOR_BGR2RGB))
  plt.title(title1)
  plt.axis('off')

  plt.subplot(1,2,1)
  plt.imshow(cv2.cvtColor(img2,cv2.COLOR_BGR2RGB))
  plt.title(title2)
  plt.axis('off')

  plt.show()
```

Importing libraries allows you to reuse pre-written code, saving time and effort when performing common tasks. Libraries provide specialized functions and tools that enhance the capabilities of your program, such as data manipulation, machine learning, or image processing. 

## **3.Load Image**

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

### **Result**

![Marj 1](https://github.com/user-attachments/assets/7a897b30-0f5a-465c-a413-539b6b8e8f4e)

## **Exercise 1:Scaling and Rotation**

```python
def scale_image(image, scale_factor):
  height, width = image.shape[:2]
  scale_image = cv2.resize(image, (int(width * scale_factor), (int(height * scale_factor))), interpolation=cv2.INTER_LINEAR)
  return scale_image

def rotated_image(image, angle):
  height, width = image.shape[:2]
  center = (width // 2, height // 2)
  matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
  rotated_image = cv2.warpAffine(image, matrix, (width, height))
  return rotated_image

scaled_image = scale_image(image, 0.5)
display_image(scaled_image, "Scaled Image (50%)")

rotated_image = rotated_image(image, 45)
display_image(rotated_image, "Rotated Image (45)")
```

### **Results**

![Marj 2](https://github.com/user-attachments/assets/1988103c-40fc-4f16-9e81-e3d6a988af41)

![Marj 3](https://github.com/user-attachments/assets/6e994349-54c5-4c57-b979-0fe974aa6b08)

## **Exercise 2: Blurring Techniques**

```python
gaussian_blur = cv2.GaussianBlur(image, (5,5), 20)
display_image(gaussian_blur, "Gaussian Blur")

median_blur = cv2.medianBlur(image, 15)
display_image(median_blur, "Median Blur")

bilateral_filter = cv2.bilateralFilter(image, 9, 75, 75)
display_image(bilateral_filter, "Belateral Filter")
```

### **Results**

![Marj 4](https://github.com/user-attachments/assets/df53d0e6-aaef-4093-a4d2-6128d342fe18)

![Marj 5](https://github.com/user-attachments/assets/dd59b7b8-0c63-4461-b054-520845971fea)

![Marj 6](https://github.com/user-attachments/assets/c7170353-cc6e-4d43-a1ea-81db524c14f7)

## **Exercise 3: Edge Detection using Canny**

```python
edges = cv2.Canny(image, 180, 180)
display_image(edges, "Canny Edge Detection")
```

### **Result**

![Marj 7](https://github.com/user-attachments/assets/5635e5c8-f226-4293-8648-5cc0fe141e13)

## **Exercise 4: Basic Image Processor (Interactive)**

```python
def process_image(img, action):
  if action == 'scale':
    return scale_image(img, 0.5)
  elif action == 'rotate':
    return rotated_image(img, 45)
  elif action == 'gaussian_blur':
    return cv2.GaussianBlur(img, (5, 5), 0)
  elif action == 'median_blur':
    return cv2.medianBlur(img, 5)
  elif action == 'canny':
    return cv2.Canny(img, 100, 200)
  else:
    return img

"""
process_image(): This function allows users to specify an image transformation (scaling, rotation, blurring, or edge detection). Depending on the action passed, it will apply the corresponding image processing technique and return the processed image.
"""
action = input("Enter action (scale, rotate, gaussian_blur, median_blur, canny: ")
processed_image = process_image(image, action)
display_images(image, processed_image, "Original Image", f"Processed Image ({action})")
"""
This allows users to enter their desired transformation interactively (via the input() function). It processes the image and displays both the original and transformed versions side by side.
"""
```

### **Result**

![Marj 8](https://github.com/user-attachments/assets/e032ace2-8c81-4b2c-bf1c-01b3f410a583)

## **Exercise 5: Comparison of Filtering Techniques**

```python
# Applying Gaussian, Median, and Bilateral filters
gaussian_blur = cv2.GaussianBlur(image, (15, 15), 0)
median_blur = cv2.medianBlur(image, 15)
bilateral_filter = cv2.bilateralFilter(image, 9, 75, 75)
"""
cv2.bilateralFilter(): This filter smooths the image while keeping edges sharp, unlike Gaussian or median filters. It’s useful for reducing noise while preserving details.
"""
# Display the results for comparison
plt.figure(figsize=(10, 5))
plt.subplot(1, 3, 1)
plt.imshow(cv2.cvtColor(gaussian_blur, cv2.COLOR_BGR2RGB))
plt.title("Gaussian Blur")
plt.subplot(1, 3, 2)
plt.imshow(cv2.cvtColor(median_blur, cv2.COLOR_BGR2RGB))
plt.title("Median Blur")
plt.subplot(1, 3, 3)
plt.imshow(cv2.cvtColor(bilateral_filter, cv2.COLOR_BGR2RGB))
plt.title("Bilateral Filter")
plt.show()

"""
Explanation: This displays the images processed by different filtering techniques (Gaussian, Median, and Bilateral) side by side for comparison.
"""
```

### **Result**

![Marj 9](https://github.com/user-attachments/assets/91c924a0-8ce2-44a5-ac0f-9343df5ed777)

## **Sobel Edge Detection**

```python
def sobel_edge_detection(img):
  # Convert to grayscale
  gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

  # Sobel edge detection in the x direction
  sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)

  # Sobel edge detection in the y direction
  sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)

  # Combine the two gradients
  sobel_combined = cv2.magnitude(sobelx, sobely)
  return sobel_combined

# Apply Sobel edge detection to the uploaded image
sobel_edges = sobel_edge_detection(image)
plt.imshow(sobel_edges, cmap='gray')
plt.title("Sobel Edge Detection")
plt.axis('off')
plt.show()
```

### **Result**

![Marj 10](https://github.com/user-attachments/assets/1f139eef-d446-42c0-a35b-57db3f385a3a)

## **Laplacian Edge Detection**

```python
def laplacian_edge_detection(img):
  # Convert to grayscale
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  # Apply Laplacian operator
  laplacian = cv2.Laplacian(gray, cv2.CV_64F)
  return laplacian
  # Apply Laplacian edge detection to the uploaded image
laplacian_edges = laplacian_edge_detection(image)
plt.imshow(laplacian_edges, cmap='gray')
plt.title("Laplacian Edge Detection")
plt.axis('off')
plt.show()
```

### **Result**

![Marj 11](https://github.com/user-attachments/assets/86bdbe18-c58c-4131-9d35-10624dac55ad)

## **Prewitt Edge Detection**

```python
def prewitt_edge_detection(img):
  # Convert to grayscale
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

  # Prewitt operator kernels for x and y directions
  kernelx = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]], dtype=int)
  kernely = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]], dtype=int)

  # Applying the Prewitt operator
  prewittx = cv2.filter2D(gray, cv2.CV_64F, kernelx)
  prewitty = cv2.filter2D(gray, cv2.CV_64F, kernely)
  # Combine the x and y gradients by converting to floating point
  prewitt_combined = cv2.magnitude(prewittx, prewitty)

  return prewitt_combined

# Apply Prewitt edge detection to the uploaded image
prewitt_edges = prewitt_edge_detection(image)
plt.imshow(prewitt_edges, cmap='gray')
plt.title("Prewitt Edge Detection")
plt.axis('off')
plt.show()
```

### **Result**

![Marj 12](https://github.com/user-attachments/assets/ce340cda-2b78-4ac6-8ab5-cd60e5c59a65)

## **Bilateral Filter**

```python
def bilateral_blur(img):
  bilateral = cv2.bilateralFilter(img, 9, 75, 75)
  return bilateral

# Apply Bilateral filter to the uploaded image
bilateral_blurred = bilateral_blur(image)
plt.imshow(cv2.cvtColor(bilateral_blurred, cv2.COLOR_BGR2RGB))
plt.title("Bilateral Filter")
plt.axis('off')
plt.show()
```

### **Result**

![Marj 13](https://github.com/user-attachments/assets/42bd1c4c-365a-43a4-9e3e-4dde7b25e6cd)

## **Box Filter**

```python
def box_blur(img):
  box = cv2.boxFilter(img, -1, (5, 5))
  return box

# Apply Box filter to the uploaded image
box_blurred = box_blur(image)
plt.imshow(cv2.cvtColor(box_blurred, cv2.COLOR_BGR2RGB))
plt.title("Box Filter")
plt.axis('off')
plt.show()
```

### **Result**

![Marj 14](https://github.com/user-attachments/assets/f34d4276-1573-4089-a957-208759894f07)

## **Motion Blur**

```python
def motion_blur(img):
  # Create motion blur kernel (size 15x15)
  kernel_size = 15
  kernel = np.zeros((kernel_size, kernel_size))
  kernel[int((kernel_size - 1) / 2), :] = np.ones(kernel_size)
  kernel = kernel / kernel_size

  # Apply motion blur
  motion_blurred = cv2.filter2D(img, -1, kernel)
  return motion_blurred

# Apply Motion blur to the uploaded image
motion_blurred = motion_blur(image)
plt.imshow(cv2.cvtColor(motion_blurred, cv2.COLOR_BGR2RGB))
plt.title("Motion Blur")
plt.axis('off')
plt.show()
```

### **Result**

![Marj 15](https://github.com/user-attachments/assets/603c01a1-0bc3-4d20-8909-c47e8142bc6a)

## **Unsharp Masking (Sharpening)**

```python
def unsharp_mask(img):
  # Create a Gaussian blur version of the image
  blurred = cv2.GaussianBlur(img, (9, 9), 10.0)

  # Sharpen by adding the difference between the original and the blurred image
  sharpened = cv2.addWeighted(img, 1.5, blurred, -0.5, 0)
  return sharpened

# Apply Unsharp Masking to the uploaded image
sharpened_image = unsharp_mask(image)
plt.imshow(cv2.cvtColor(sharpened_image, cv2.COLOR_BGR2RGB))
plt.title("Unsharp Mask (Sharpening)")
plt.axis('off')
plt.show()
```

### **Result**

![Marj 16](https://github.com/user-attachments/assets/c12ab0cf-830e-47f0-9da1-28a748929902)

# **Update process_image function to include new blurring techniques**

```python
def process_image(img, action):
  if action == 'scale':
    return scale_image(img, 0.5)
  elif action == 'rotate':
    return rotate_image(img, 45)
  elif action == 'gaussian_blur':
    return cv2.GaussianBlur(img, (5, 5), 0)
  elif action == 'median_blur':
    return cv2.medianBlur(img, 5)
  elif action == 'canny':
    return cv2.Canny(img, 100, 200)
  elif action == 'sobel':
    return sobel_edge_detection(img)
  elif action == 'laplacian':
    return laplacian_edge_detection(img)
  elif action == 'prewitt':
    return prewitt_edge_detection(img)
  elif action == 'bilateral_blur':
    return bilateral_blur(img)
  elif action == 'box_blur':
    return box_blur(img)
  elif action == 'motion_blur':
    return motion_blur(img)
  elif action == 'unsharp_mask':
    return unsharp_mask(img)
  else:
    return img

# Add new blurring options for interactive processing
action = input("Enter action (scale, rotate, gaussian_blur, median_blur, canny, sobel, laplacian, prewitt, bilateral_blur, box_blur, motion_blur")
processed_image = process_image(image, action)
display_images(image, processed_image, "Original Image", f"Processed Image ({action})")
```

![Marj 17](https://github.com/user-attachments/assets/42a947e0-9c45-4c7d-8aa9-84d4e31d6447)

# **Comparison**

```python
# Original Image
display_image(image, "Original Image")

# Blurring
plt.figure(figsize=(10, 5))
plt.subplot(1, 3, 1)
plt.imshow(cv2.cvtColor(gaussian_blur, cv2.COLOR_BGR2RGB))
plt.title("Gaussian Blur")
plt.subplot(1, 3, 2)
plt.imshow(cv2.cvtColor(median_blur, cv2.COLOR_BGR2RGB))
plt.title("Median Blur")
plt.subplot(1, 3, 3)
plt.imshow(cv2.cvtColor(bilateral_filter, cv2.COLOR_BGR2RGB))
plt.title("Bilateral Filter")
plt.show()

plt.figure(figsize=(10, 5))
plt.subplot(1, 3, 1)
plt.imshow(cv2.cvtColor(box_blurred, cv2.COLOR_BGR2RGB))
plt.title("Box Blur")
plt.subplot(1, 3, 2)
plt.imshow(cv2.cvtColor(motion_blurred, cv2.COLOR_BGR2RGB))
plt.title("Motion Blur")
plt.subplot(1, 3, 3)
plt.imshow(cv2.cvtColor(sharpened_image, cv2.COLOR_BGR2RGB))
plt.title("Unsharp Mask")
plt.show()

# Edge Detection
sobel_edges_uint8 = cv2.convertScaleAbs(sobel_edges)
laplacian_edges_uint8 = cv2.convertScaleAbs(laplacian_edges)
prewitt_edges_uint8 = cv2.convertScaleAbs(prewitt_edges)
edges_uint8 = edges

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(sobel_edges_uint8, cv2.COLOR_BGR2RGB))
plt.title("Sobel Edge Detection")
plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(laplacian_edges_uint8, cv2.COLOR_BGR2RGB))
plt.title("Laplacian Edge Detection")
plt.show()

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(prewitt_edges_uint8, cv2.COLOR_BGR2RGB))
plt.title("Prewitt Edge Detection")
plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(edges_uint8, cv2.COLOR_BGR2RGB))
plt.title("Canny Edge Detection")
plt.show()
```

![image](https://github.com/user-attachments/assets/81ca3cf9-190c-4155-b9fd-c5bca9042f46)
![image](https://github.com/user-attachments/assets/7d4e0bad-2d80-433b-822d-1a5637e18607)
![image](https://github.com/user-attachments/assets/ff45c23a-4568-4f2b-8edc-c29a934bdf06)


***Link to the Google Colab Project:***

* [EXERCISE1-ImageProcessingTechniques-CAYADONG-4B.ipynb](https://colab.research.google.com/drive/1vtBP1-7w41lG-nQ_BfSeE3_9AzaW8G5V?usp=sharing)

*Comparison**

* [4B-CAYADONG-EXER1.pdf](https://github.com/user-attachments/files/17062910/4B-CAYADONG-EXER1.pdf)

