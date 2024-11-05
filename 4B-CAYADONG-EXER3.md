# **Exer3 - Advanced Feature Extraction and Image Processing**

![EXER 3](https://github.com/user-attachments/assets/6da6a5f1-a1dc-4080-ab60-bb1ab5c6a820)

**Install Libraries**

```python
!pip install opencv-python-headless
```

**Import Libraries**
```python
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import hog
from skimage import exposure
```

## **Exercise 1: Harris Corner Detection**

**Load an image of your choice**

```python
img1 = cv2.imread('uhh.jpg')
```

**Convert it to grayscale**

```python
gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray = np.float32(gray)
```

**Apply the Harris Corner Detection method to detect corners**

```python
harris = cv2.cornerHarris(gray, blockSize=2, ksize=3, k=0.04)
harris = cv2.dilate(harris, None)
img_with_corners = img1.copy()
img_with_corners[harris > 0.01 * harris.max()] = [0, 0, 255]
corner_harris = cv2.cvtColor(img_with_corners, cv2.COLOR_BGR2RGB)
```

**Visualize the corners on the image and display the result**

```python
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(corner_harris)
plt.title('Harris Corner Detection')
plt.axis('off')

plt.show()
```
![image1](https://github.com/user-attachments/assets/e7bec222-5e30-4ffe-84ad-3937134abae7)

## **Exercise 2: HOG (Histogram of Oriented Gradients) Feature Extraction**

**Load an image of a person or any object.**

```python
img = cv2.imread('strawberry_c.jpg')
```

**Convert the image to grayscale**

```python
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
```

**Apply the HOG descriptor to extract features**

```python
hog_ft, hog_img = hog(gray, orientations=9, pixels_per_cell=(8, 8),
                              cells_per_block=(2, 2), visualize=True, block_norm='L2-Hys')

hog_img_res = exposure.rescale_intensity(hog_img, in_range=(0, 10))
```

**Visualize the gradient orientations on the image.**

```python
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(hog_img_res, cmap='gray')
plt.title('HOG Visualization')
plt.axis('off')

plt.show()
```
![image2](https://github.com/user-attachments/assets/04849c76-e899-48cd-a63a-553d9f8c2719)


## **Exercise 3: FAST (Features from Accelerated Segment Test) Keypoint Detection**

**Load an image**

```python
img = cv2.imread('green.jpg')
```

**Convert the image to grayscale**

```python
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
```

**Apply the FAST algorithm to detect keypoints**

```python
fast = cv2.FastFeatureDetector_create()
keypoints = fast.detect(gray, None)
img_with_kps = cv2.drawKeypoints(img, keypoints, None, color=(0, 255, 0))
fast_kpd = cv2.cvtColor(img_with_kps, cv2.COLOR_BGR2RGB)
```

**Visualize the keypoints on the image and display the result**

```python
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(fast_kpd)
plt.title('FAST Keypoint Detection')
plt.axis('off')

plt.show()
```
![image3](https://github.com/user-attachments/assets/b817f7e9-f833-4351-bc48-0fd50a6293e8)

### **Google Colab Link:**

* [EXERCISE3-AdvancedFeatureExtractionandImageProcessing-CAYADONG-4B.ipynb](https://colab.research.google.com/drive/13EvPN9wKMH3lQ6C5gDg6oY-vWlFmtDMN?usp=sharing)
