# **Exercise-2-Feature Extraction Methods**

![EXER 2](https://github.com/user-attachments/assets/e6385a9a-d14b-4b69-b819-57e093bd5da5)

# **Feature Extraction**

Feature extraction is a process in machine learning and data analysis that involves identifying and extracting relevant features from raw data. These features are later used to create a more informative dataset, which can be further utilized for various tasks such as:

* Classification

* Prediction

* Clustering

Feature extraction aims to reduce data complexity (often known as “data dimensionality”) while retaining as much relevant information as possible. This helps to improve the performance and efficiency of machine learning algorithms and simplify the analysis process. Feature extraction may involve the creation of new features (“feature engineering”) and data manipulation.

## *Why is Feature Extraction Important?*

  Feature extraction plays a vital role in many real-world applications. Feature extraction is critical for processes such as image and speech recognition, predictive modeling, and Natural Language Processing (NLP). In these scenarios, the raw data may contain many irrelevant or redundant features. This makes it difficult for algorithms to accurately process the data.

  Feature extraction plays an important role in image processing. This technique is used to detect features in digital images such as edges, shapes, or motion. Once these are identified, the data can be processed to perform various tasks related to analyzing an image. 

## **Advantages and Importance**

Feature extraction methods are instrumental in enhancing the interpretability, efficiency, and accuracy of AI models. By discerning relevant patterns and minimizing data redundancy, these methods contribute to streamlined decision-making processes, resource optimization, and model generalization.

Influence on AI Applications The significance of feature extraction methods extends across various domains of AI applications, including computer vision, natural language processing, sensor data analysis, and bioinformatics. Their utility in extracting salient features from diverse data types has revolutionized the capability of AI systems to comprehend and analyze complex information.

# **Overview of image matching techniques**

### **SIFT**

* SIFT proposed by Lowe solves the image rotation, affine transformations, intensity, and viewpoint change in matching features. The SIFT algorithm has 4 basic steps. First is to estimate a scale space extrema using the Difference of Gaussian (DoG). Secondly, a key point localization where the key point candidates are localized and refined by eliminating the low contrast points. Thirdly, a key point orientation assignment based on local image gradient and lastly a descriptor generator to compute the local image descriptor for each key point based on image gradient magnitude and orientation.

### **SURF**

* SURF approximates the DoG with box filters. Instead of Gaussian averaging the image, squares are used for approximation since the convolution with square is much faster if the integral image is used. Also this can be done in parallel for different scales. The SURF uses a BLOB detector which is based on the Hessian matrix to find the points of interest. For orientation assignment, it uses wavelet responses in both horizontal and vertical directions by applying adequate Gaussian weights. For feature description also SURF uses the wavelet responses. A neighborhood around the key point is selected and divided into subregions and then for each subregion the wavelet responses are taken and represented to get SURF feature descriptor. The sign of Laplacian which is already computed in the detection is used for underlying interest points. The sign of the Laplacian distinguishes bright blobs on dark backgrounds from the reverse case. In case of matching the features are compared only if they have same type of contrast (based on sign) which allows faster matching.

### **ORB**

* ORB is a fusion of the FAST key point detector and BRIEF descriptor with some modifications [9]. Initially to determine the key points, it uses FAST. Then a Harris corner measure is applied to find top N points. FAST does not compute the orientation and is rotation variant. It computes the intensity weighted centroid of the patch with located corner at center. The direction of the vector from this corner point to centroid gives the orientation. Moments are computed to improve the rotation invariance. The descriptor BRIEF poorly per forms if there is an in-plane rotation. In ORB, a rotation matrix is computed using the orientation of patch and then the BRIEF descriptors are steered according to the orientation.

#

***Links and Sources used in this study:***

* https://domino.ai/data-science-dictionary/feature-extraction

* https://www.snowflake.com/guides/feature-extraction-machine-learning/#:~:text=Feature%20extraction%20plays%20an%20important,related%20to%20analyzing%20an%20image.

* https://www.larksuite.com/en_us/topics/ai-glossary/some-common-methods-for-feature-extraction-in-ai

* https://shehan-a-perera.medium.com/a-comparison-of-sift-surf-and-orb-333d64bcaaea#:~:text=So%20we%20can%20notice%20that,ORB%20execute%20fast%20than%20others.
