# **Machine Problem 4 - Feature Extraction and Image Matching in Computer Vision**

![MP 4](https://github.com/user-attachments/assets/0c7c2dc2-bb7e-4ffc-856f-a66626ae30f9)

***Link to the Google Colab Project:***

[4B-CAYADONG-MP4.ipynb](https://colab.research.google.com/drive/1sGEvxeELX9DcOrshhgXMCBoNfKFy69HZ?usp=sharing)

**What Is Feature Extraction?**

Feature extraction refers to the process of transforming raw data into numerical features that can be processed while preserving the information in the original data set. It yields better results than applying machine learning directly to the raw data.Manual feature extraction requires identifying and describing the features that are relevant for a given problem and implementing a way to extract those features. In many situations, having a good understanding of the background or domain can help make informed decisions as to which features could be useful. 

## **Harris Corner Detection**

Harris Corner Detector is a corner detection operator that is commonly used in computer vision algorithms to extract corners and infer features of an image. Harris’ corner detector takes the differential of the corner score into account with reference to direction directly, instead of using shifting patches for every 45-degree angles, and has been proved to be more accurate in distinguishing between edges and corners.

```python
def harris_corner_detection(image_path):
    img1 = cv2.imread('/content/niii.jpg')
    gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)

    corners = cv2.cornerHarris(gray, 2, 3, 0.04)
    corners = cv2.dilate(corners, None)
    img1[corners > 0.01 * corners.max()] = [0, 0, 255]

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(cv2.imread('/content/niii.jpg'), cv2.COLOR_BGR2RGB))
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
    plt.title('Harris Corner Detection')
    plt.axis('off')

    plt.show()
```

*Result*

![image1](https://github.com/user-attachments/assets/c0ab072d-d3a6-4b26-9704-94e09618e2a9)

## **HOG Feature Extraction**

Histogram of Oriented Gradients (HOG) is a powerful feature extraction technique that is extremely useful for medical image analysis. In this blog, I will deep dive into how HOG can be used as a Feature extractor for Images and what makes HOG an effective tool for extracting features from medical images. HOG is a feature descriptor that counts occurrences of gradient orientation in localized portions of an image. 

```python
def hog_feature_extraction(image_path):
    img = cv2.imread('/content/niii.jpg')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    features, hog_image = hog(gray, pixels_per_cell=(8, 8), cells_per_block=(2, 2),
                              visualize=True, channel_axis=None) 

    hog_image = exposure.rescale_intensity(hog_image, in_range=(0, 10))

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(hog_image, cmap='gray')
    plt.title('HOG Features')
    plt.axis('off')
    plt.show()
```

*Result*

![image2](https://github.com/user-attachments/assets/2baadc50-ada3-41ad-b2af-cb0165700fbc)

## **ORB Feature Extraction and Matching**

ORB is a fusion of FAST keypoint detector and BRIEF descriptor with some added features to improve the performance. FAST is Features from Accelerated Segment Test used to detect features from the provided image. It also uses a pyramid to produce multiscale-features. ORB uses BRIEF descriptors but as the BRIEF performs poorly with rotation. So what ORB does is to rotate the BRIEF according to the orientation of keypoints. Using the orientation of the patch, its rotation matrix is found and rotates the BRIEF to get the rotated version.

```python
def orb_feature_matching(image_path1, image_path2):
    img1 = cv2.imread('/content/lappy.jpg', cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread('/content/niii.jpg', cv2.IMREAD_GRAYSCALE)

    orb = cv2.ORB_create()
    keypoints1, descriptors1 = orb.detectAndCompute(img1, None)
    keypoints2, descriptors2 = orb.detectAndCompute(img2, None)

    if descriptors1 is None or descriptors2 is None:
        print("No descriptors found in one of the images. Cannot match features.")
        return

    if descriptors1.shape[1] != descriptors2.shape[1]:
        print("Descriptor dimensions do not match. Cannot perform matching.")
        return

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(descriptors1, descriptors2)
    matches = sorted(matches, key=lambda x: x.distance)

    img_matches = cv2.drawMatches(img1, keypoints1, img2, keypoints2, matches[:20], None, flags=2)

    plt.figure(figsize=(12, 6))
    plt.imshow(img_matches)
    plt.title('ORB Feature Matching')
    plt.axis('off')
    plt.show()
```

*Result*

![image3](https://github.com/user-attachments/assets/55958c9a-73a5-4685-86d9-e0a5685291f9)

## **SIFT and SURF Feature Extraction**

These two robust feature descriptors are invariant to scale changes, blur, rotation, illumination changes and affine transformation. SIFT is an algorithm used to extract the features from the images. SURF is an efficient algorithm is same as SIFT performance and reduced in computational complexity. SIFT algorithm presents its ability in most of the situation but still its performance is slow. SURF algorithm is same as SIFT with fastest one and good performance.

```python
def sift_and_surf_feature_extraction(image_path1, image_path2):
    img1 = cv2.imread('/content/niii.jpg')
    img2 = cv2.imread('/content/lappy.jpg')

    sift = cv2.SIFT_create()
    keypoints1_sift, _ = sift.detectAndCompute(img1, None)
    keypoints2_sift, _ = sift.detectAndCompute(img2, None)
    img1_sift = cv2.drawKeypoints(img1, keypoints1_sift, None)
    img2_sift = cv2.drawKeypoints(img2, keypoints2_sift, None)

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(img1_sift, cv2.COLOR_BGR2RGB))
    plt.title('SIFT Keypoints Image 1')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(img2_sift, cv2.COLOR_BGR2RGB))
    plt.title('SIFT Keypoints Image 2')
    plt.axis('off')
    plt.show()
```

*Result*

![image4](https://github.com/user-attachments/assets/fc333f18-3be3-4635-8726-ba3755986aba)

## **Feature Matching using Brute-Force Matcher**

Feature matching using a brute-force matcher is a computer vision technique that compares feature points in two images by matching each feature in one image with every feature in the other image. The closest match is then returned.

```python
def brute_force_feature_matching(image_path1, image_path2):
    img1 = cv2.imread('/content/niii.jpg', cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread('/content/lappy.jpg', cv2.IMREAD_GRAYSCALE)

    orb = cv2.ORB_create()
    keypoints1, descriptors1 = orb.detectAndCompute(img1, None)
    keypoints2, descriptors2 = orb.detectAndCompute(img2, None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(descriptors1, descriptors2)
    matches = sorted(matches, key=lambda x: x.distance)

    img_matches = cv2.drawMatches(img1, keypoints1, img2, keypoints2, matches[:20], None, flags=2)

    plt.figure(figsize=(12, 6))
    plt.imshow(img_matches)
    plt.title('Brute-Force Feature Matching')
    plt.axis('off')
    plt.show()
```

*Result*

![image5](https://github.com/user-attachments/assets/45b888d3-0ddd-4dd8-bf04-0d37ae080efc)

## **Image Segmentation using Watershed Algorithm**

Watershed Segmentation is a region-based method in computer science that treats an image as a topographic landscape, separating it into catchment basins based on pixel values or gradients, and is commonly used in medical image segmentation tasks. 

```python
def watershed_segmentation(image_path):
    img = cv2.imread('/content/lappy.jpg')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)

    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)

    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)

    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0

    markers = cv2.watershed(img, markers)
    img[markers == -1] = [255, 0, 0]

    plt.figure(figsize=(12, 6))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title('Watershed Segmentation')
    plt.axis('off')
    plt.show()
```

*Result*

![image6](https://github.com/user-attachments/assets/bf8e5452-5cf7-454e-bb11-dc3ac920853f)

Feature extraction is a critical technique in data processing that focuses on transforming raw data into numerical features that retain essential information for machine learning models. It allows for more effective model training by highlighting relevant patterns or structures in data, like image details, while discarding unnecessary noise.

***Importance of Feature Extraction in Image Processing***

These feature extraction techniques are foundational in computer vision, as they allow models to identify patterns, objects, and structures efficiently. By extracting important features, machine learning models gain a clear and simplified view of the input data, enabling them to make more accurate predictions. In fields like medical imaging, autonomous driving, and security, these methods are essential for tasks like object detection, segmentation, and image matching.

***Links used in this documentation:***

* https://in.mathworks.com/discovery/feature-extraction.html#:~:text=What%20Is%20Feature%20Extraction%3F,directly%20to%20the%20raw%20data.
* https://medium.com/@deepanshut041/introduction-to-harris-corner-detector-32a88850b3f6
* https://medium.com/@girishajmera/hog-histogram-of-oriented-gradients-an-amazing-feature-extraction-engine-for-medical-images-5a2203b47ccd#:~:text=Histogram%20of%20Oriented%20Gradients%20(HOG,extracting%20features%20from%20medical%20images.
* https://geeksforgeeks.org/feature-matching-using-orb-algorithm-in-python-opencv/
* https://ieeexplore.ieee.org/document/7975187
