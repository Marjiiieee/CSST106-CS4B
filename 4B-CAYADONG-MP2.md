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

