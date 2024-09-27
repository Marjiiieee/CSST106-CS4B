"""
# **1. Install Necessary Libraries for SIFT, SURF and ORB**
"""

!pip uninstall -y opencv-python opencv-python-headless opencv-contrib-python

!apt-get install -y cmake
!apt-get install -y libopencv-dev build-essential cmake git pkg-config libgtk-3-dev \
   libavcodec-dev libavformat-dev libswscale-dev libtbb2 libtbb-dev libjpeg-dev \
   libpng-dev libtiff-dev libdc1394-22-dev libv4l-dev v4l-utils \
   libxvidcore-dev libx264-dev libxine2-dev gstreamer1.0-tools \
   libgstreamer-plugins-base1.0-dev libgstreamer-plugins-good1.0-dev \
   libgtk2.0-dev libtiff5-dev libopenexr-dev libatlas-base-dev \
   python3-dev python3-numpy libtbb-dev libeigen3-dev \
   libfaac-dev libmp3lame-dev libtheora-dev libvorbis-dev \
   libxvidcore-dev libx264-dev yasm libopencore-amrnb-dev \
   libopencore-amrwb-dev libv4l-dev libxine2-dev libtesseract-dev \
   liblapacke-dev libopenblas-dev checkinstall

!git clone https://github.com/opencv/opencv.git
!git clone https://github.com/opencv/opencv_contrib.git

# Commented out IPython magic to ensure Python compatibility.
# %cd opencv
!mkdir build
# %cd build

!cmake -D CMAKE_BUILD_TYPE=RELEASE \
       -D CMAKE_INSTALL_PREFIX=/usr/local \
       -D OPENCV_ENABLE_NONFREE=ON \
       -D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib/modules \
       -D BUILD_EXAMPLES=ON ..

!make -j8
!make install

!pip install opencv-python-headless
!pip install opencv-contrib-python

"""## **Step 1: Load Images**"""

from skimage import io, color
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load two images
image1 = cv2.imread('Front.jpg')  # First image
image2 = cv2.imread('Side.jpg')  # Second image

# Convert images to grayscale
gray_image1 = color.rgb2gray(image1)
gray_image2 = color.rgb2gray(image2)

gray_image1 = (gray_image1 * 255).astype(np.uint8)
gray_image2 = (gray_image2 * 255).astype(np.uint8)

# Display images
plt.figure(figsize=(5, 5))
plt.imshow(image1)
plt.title('Grayscale Image 1')
plt.axis('off')

plt.figure(figsize=(5, 5))
plt.imshow(image2)
plt.title('Grayscale Image 2')
plt.axis('off')

plt.show()

"""## **Step 2: Extract Keypoints and Descriptors Using SIFT, SURF, and ORB**"""

#SIFT
sift = cv2.SIFT_create()
keypoints1_sift, descriptors1_sift = sift.detectAndCompute(gray_image1, None)
keypoints2_sift, descriptors2_sift = sift.detectAndCompute(gray_image2, None)

print(f"SIFT - Image 1: {len(keypoints1_sift)} keypoints detected")
print(f"SIFT - Image 2: {len(keypoints2_sift)} keypoints detected")

#SURF
surf = cv2.xfeatures2d.SURF_create(hessianThreshold=400)
keypoints1_surf, descriptors1_surf = surf.detectAndCompute(gray_image1, None)
keypoints2_surf, descriptors2_surf = surf.detectAndCompute(gray_image2, None)

print(f"SURF - Image 1: {len(keypoints1_surf)} keypoints detected")
print(f"SURF - Image 2: {len(keypoints2_surf)} keypoints detected")

# ORB
orb = cv2.ORB_create()
keypoints1_orb, descriptors1_orb = orb.detectAndCompute(gray_image1, None)
keypoints2_orb, descriptors2_orb = orb.detectAndCompute(gray_image2, None)

print(f"ORB - Image 1: {len(keypoints1_orb)} keypoints detected")
print(f"ORB - Image 2: {len(keypoints2_orb)} keypoints detected")

# Draw keypoints for SIFT
image1_sift = cv2.drawKeypoints(image1, keypoints1_sift, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
image2_sift = cv2.drawKeypoints(image2, keypoints2_sift, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# Draw keypoints for SURF
image1_surf = cv2.drawKeypoints(image1, keypoints1_surf, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
image2_surf = cv2.drawKeypoints(image2, keypoints2_surf, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# Draw keypoints for ORB
image1_orb = cv2.drawKeypoints(image1, keypoints1_orb, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
image2_orb = cv2.drawKeypoints(image2, keypoints2_orb, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

"""**Visuals**"""

import matplotlib.pyplot as plt

# Display the images with keypoints using Matplotlib
plt.figure(figsize=(15, 10))

# Display SIFT keypoints
plt.subplot(3, 2, 1)
plt.imshow(image1_sift)
plt.title('SIFT Keypoints - Image 1')
plt.axis('off')
plt.subplot(3, 2, 2)
plt.imshow(image2_sift)
plt.title('SIFT Keypoints - Image 2')
plt.axis('off')

# Display SURF keypoints
plt.subplot(3, 2, 3)
plt.imshow(image1_surf)
plt.title('SURF Keypoints - Image 1')
plt.axis('off')
plt.subplot(3, 2, 4)
plt.imshow(image2_surf)
plt.title('SURF Keypoints - Image 2')
plt.axis('off')

# Display ORB keypoints
plt.subplot(3, 2, 5)
plt.imshow(image1_orb)
plt.title('ORB Keypoints - Image 1')
plt.axis('off')
plt.subplot(3, 2, 6)
plt.imshow(image2_orb)
plt.title('ORB Keypoints - Image 2')
plt.axis('off')

# Show the plot
plt.tight_layout()
plt.show()
