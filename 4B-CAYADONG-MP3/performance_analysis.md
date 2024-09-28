# **Performance Analysis**

**Comparison of Results**

##### **Keypoint Detection Accuracy and Number of Keypoints**

- **SIFT**:
  - It has high accuracy in detecting keypoints, especially for images with varying scales and rotation. SIFT is strong and detects distinctive keypoints even under slight transformations.
  - It detects a moderate number of keypoints, often higher than ORB but lower than SURF.
  - SIFT is almost as fast as ORB, it doesn't need other long processed libraries for it to function compared to ORB and SURF because it computes detailed descriptors for every keypoint.
 
  - ![image](https://github.com/user-attachments/assets/5d1a8e06-46b4-4cfc-9cfe-2ec13300229c)

- **SURF**:
  - Similar to SIFT in terms of scale and rotation invariance, but slower due to the use of approximations. Slightly less accurate than SIFT for fine details but still very effective.
  - SURF tends to detect more keypoints compared to SIFT and ORB, especially in textured or complex images.
  - Faster than SIFT, but slower on downloading other libraries in need to make it function, but it makes a good balance between accuracy and performance for real-time applications.
 
  - ![image](https://github.com/user-attachments/assets/4ca7ea1e-a0f9-4800-8ae0-9832c6a0e02c)

- **ORB**:
  - ORB is a fast alternative to SIFT and SURF but is less powerful to scale changes and image transformations. ORB works well for straightforward matching tasks but struggles in cases where the viewpoint changes, such as rotating it.
  - ORB detects a large number of keypoints, usually more than SIFT but with less precision and stability.
  - ORB is the fastest of the three.
 
  - ![image](https://github.com/user-attachments/assets/b330fd6a-4e06-471a-95c7-e95ba982acab)

##### **Comments**

1. **SIFT (Scale-Invariant Feature Transform)**:
   - SIFT provides excellent accuracy and is powerful to scale, rotation, and minor viewpoint changes.
   - It detects a moderate number of keypoints, which makes it efficient in terms of memory usage and processing time.
   - The major downside of SIFT is its a bit slower than ORB, especially for larger images, but its accuracy compensates for this in applications requiring precise feature detection.
  
     ![sift_keypoints](https://github.com/user-attachments/assets/b5d91b67-0bc8-4c31-b8e0-6e02ab5bd71f)

2. **SURF (Speeded-Up Robust Features)**:
   - SURF offers a good balance between accuracy and speed (When using it). It detects more keypoints than SIFT and handles scale and rotation similarly well.
   - It is faster than SIFT due to the use of approximations but slightly less accurate in detecting fine details.
   - SURF is effective in real-time applications where speed is critical, though it may not be as precise in highly detailed or low-contrast images.
   - SURF has a downside to not having a pre-installed library or restriction to other programing language (google colaboratory), making it very difficult to re-process the data needed to use it, it can take up at least 1 - 3 hours depending on the internet speed.
  
     ![surf_keypoints](https://github.com/user-attachments/assets/8f940424-b079-485c-bc28-99cb6c1f234b)

3. **ORB (Oriented FAST and Rotated BRIEF)**:
   - ORB is the fastest algorithm among the three. It is suitable for scenarios requiring real-time processing but sacrifices accuracy in complex scenes or cases with significant scale or rotation changes.
   - While it detects a large number of keypoints, they are not as distinctive as those detected by SIFT or SURF, leading to more false matches.
  
     ![orb_keypoints](https://github.com/user-attachments/assets/955989ff-67a0-49ec-948e-7fadaf506e0f)

**Feature Matching**:
   - **Brute-Force Matcher** worked well for all three algorithms but was noticeably slower for SIFT and SURF, particularly on larger datasets. For ORB, it is better to use Brute-Force Matcher.

![sift_bf_match](https://github.com/user-attachments/assets/2154d2db-b551-421a-8883-ad2e34a1ead9)

![surf_bf_match](https://github.com/user-attachments/assets/ec347cb5-bb4d-4012-b095-09e6c33dca33)

![orb_bf_match](https://github.com/user-attachments/assets/4dc931b1-3df0-429e-81bb-fd124daaa704)
     
   - **FLANN Matcher** improved the speed of SIFT and SURF matching by reducing the search space using approximate nearest neighbors, without significantly sacrificing accuracy. FLANN is not the best choice for ORB because it is not optimized for the binary format that ORB uses.

![sift_flann_match](https://github.com/user-attachments/assets/d92e87be-8aa4-4872-875c-d717dc868980)

![surf_flann_match](https://github.com/user-attachments/assets/52756dc4-e58a-4483-ac6b-9028fa0b9e5b)

![orb_flann_match](https://github.com/user-attachments/assets/70c4bf78-c6d4-4424-b0fe-7edbde43dd4d)

# ***Link to the Google Colab Project***

* [4B-CAYADONG-MP3.ipynb](https://colab.research.google.com/drive/1KMEcwa-Z6tAn3d1A0oCe7l8_v6BkeWvW?usp=sharing)
