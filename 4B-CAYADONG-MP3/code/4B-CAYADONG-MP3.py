# -*- coding: utf-8 -*-
"""4B-CAYADONG-MP3.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1KMEcwa-Z6tAn3d1A0oCe7l8_v6BkeWvW

# **CSST 106** - Perception and Computer Vision

**Name:** Cayadong, Marjelaine M.

**Program, Year & Section:** BSCS - 4B

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

"""## **Step 3: Feature Matching with Brute-Force and FLANN**"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

# Brute-Force Matching with SIFT Descriptors
def bf_matcher(descriptors1, descriptors2, norm_type):
    bf = cv2.BFMatcher(norm_type, crossCheck=True)
    matches = bf.match(descriptors1, descriptors2)
    matches = sorted(matches, key=lambda x: x.distance)
    return matches

# FLANN Based Matching with SIFT Descriptors
def flann_matcher(descriptors1, descriptors2):
    index_params = dict(algorithm=1, trees=5)
    search_params = dict(checks=50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(descriptors1, descriptors2, k=2)

    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)
    return good_matches

# Draw matches
def draw_matches(image1, keypoints1, image2, keypoints2, matches, title):
    result_img = cv2.drawMatches(image1, keypoints1, image2, keypoints2, matches[:50], None, flags=2)
    plt.figure(figsize=(10, 7))
    plt.imshow(result_img)
    plt.title(title)
    plt.axis('off')
    plt.show()

# Feature Matching using Brute-Force Matcher for SIFT
bf_matches_sift = bf_matcher(descriptors1_sift, descriptors2_sift, norm_type=cv2.NORM_L2)
draw_matches(image1, keypoints1_sift, image2, keypoints2_sift, bf_matches_sift, "Brute-Force Matcher SIFT")

# Feature Matching using FLANN Matcher for SIFT
flann_matches_sift = flann_matcher(descriptors1_sift, descriptors2_sift)
draw_matches(image1, keypoints1_sift, image2, keypoints2_sift, flann_matches_sift, "FLANN Matcher SIFT")

# Repeat Brute-Force Matcher for SURF
bf_matches_surf = bf_matcher(descriptors1_surf, descriptors2_surf, norm_type=cv2.NORM_L2)
draw_matches(image1, keypoints1_surf, image2, keypoints2_surf, bf_matches_surf, "Brute-Force Matcher SURF")

# Repeat FLANN Matcher for SURF
flann_matches_surf = flann_matcher(descriptors1_surf, descriptors2_surf)
draw_matches(image1, keypoints1_surf, image2, keypoints2_surf, flann_matches_surf, "FLANN Matcher SURF")

# Repeat Brute-Force Matcher for ORB
bf_matches_orb = bf_matcher(descriptors1_orb, descriptors2_orb, norm_type=cv2.NORM_HAMMING)
draw_matches(image1, keypoints1_orb, image2, keypoints2_orb, bf_matches_orb, "Brute-Force Matcher ORB")

# Repeat FLANN Matcher for ORB
index_params_orb = dict(algorithm=6, table_number=6, key_size=12, multi_probe_level=1)  # FLANN_INDEX_LSH
search_params_orb = dict(checks=50)

flann_orb = cv2.FlannBasedMatcher(index_params_orb, search_params_orb)
matches_orb = flann_orb.knnMatch(descriptors1_orb, descriptors2_orb, k=2)

good_matches_orb = []
for m, n in matches_orb:
    if m.distance < 0.7 * n.distance:
        good_matches_orb.append(m)

draw_matches(image1, keypoints1_orb, image2, keypoints2_orb, good_matches_orb, "FLANN Matcher ORB")

"""## **Step 4: Image Alignment Using Homography**"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

def align_images(image1, image2, keypoints1, keypoints2, matches):
    # Extract location of good matches
    src_pts = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    # Compute the homography matrix
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    # Get the shape of the second image
    h, w, _ = image1.shape
    h, w, _ = image2.shape

    # Warp the images to align with each image
    warped_image1 = cv2.warpPerspective(image1, H, (w, h))
    warped_image2 = cv2.warpPerspective(image2, H, (w, h))

    return warped_image1, H
    return warped_image2, H

warped_image1, homography_matrix = align_images(image1, image2, keypoints1_sift, keypoints2_sift, bf_matches_sift)
warped_image2, homography_matrix = align_images(image1, image2, keypoints1_sift, keypoints2_sift, bf_matches_sift)

# Display the warped images
plt.figure(figsize=(10, 7))

# Show warped image and original image of 1 and 2
plt.subplot(2, 2, 1)
plt.imshow(cv2.cvtColor(warped_image2, cv2.COLOR_BGR2RGB))
plt.title('Warped Image 2')
plt.axis('off')
plt.subplot(2, 2, 2)
plt.imshow(cv2.cvtColor(image1, cv2.COLOR_BGR2RGB))
plt.title('Original Image 1')
plt.axis('off')

# Show warped image and original image of 2 and 1
plt.subplot(2, 2, 3)
plt.imshow(cv2.cvtColor(warped_image1, cv2.COLOR_BGR2RGB))
plt.title('Warped Image 1')
plt.axis('off')
plt.subplot(2, 2, 4)
plt.imshow(cv2.cvtColor(image2, cv2.COLOR_BGR2RGB))
plt.title('Original Image 2')
plt.axis('off')

plt.show()