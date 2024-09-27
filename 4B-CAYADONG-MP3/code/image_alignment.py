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