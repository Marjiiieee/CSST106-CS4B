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