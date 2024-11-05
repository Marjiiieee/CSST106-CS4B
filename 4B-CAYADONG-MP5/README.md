# **Machine Problem 5 - Object Detection and Recognition using YOLO (Documentation)**

![MP 5](https://github.com/user-attachments/assets/0f5e0b22-e71f-42d4-bdc8-fbbcab7e2f7c)

***Link to the Google Colab Project:***

[4B-CAYADONG-MP5.ipynb](https://colab.research.google.com/drive/1YVEmaccIk3yaxstFr7ggzwboFPNOKKX5?usp=sharing)

**What is Object Detection and Recognition using Yolo?**

Computer vision is a rapidly advancing field that aims to enable computers to interpret and understand visual information in the same way that humans do. One of the most important subfields of computer vision is object detection, which involves detecting instances of semantic objects of a certain pre-defined class(es), such as humans, cars, and animals, in images or frames of videos.

YOLO (You Only Look Once) is a real-time object detection algorithm known for its speed and accuracy. YOLO treats object detection as a single regression problem, directly predicting bounding boxes and class probabilities from the entire image in a single evaluation. This approach contrasts with older methods that scan the image multiple times, making YOLO much faster and efficient, especially useful for real-time applications.

```python
import cv2
import time
import matplotlib.pyplot as plt
from ultralytics import YOLO
```

This code imports libraries to set up a YOLO-based object detection pipeline. It uses OpenCV for image handling, time for performance tracking, and Matplotlib for displaying results. The ultralytics library allows easy loading and running of YOLO models for object detection on images or video streams.

```python
model = YOLO('yolov8n.pt')

image_paths = ['cats1.jpg', 'cats2.jpg', 'cats3.jpg']
labels = ['Cats 1', 'Cats 2', 'Cats 3']

explain this code in 4 sentences
```

This code initializes a YOLO model using the pre-trained yolov8n.pt file, which contains a lightweight YOLOv8 model optimized for efficient object detection. The image_paths list stores the file paths of images that will be processed by the model, while the labels list holds corresponding labels for each image. This setup enables easy pairing of each image with its label, allowing the user to keep track of or display which image corresponds to which label during processing. Together, this forms the foundation for running object detection on multiple images and associating them with custom labels for clear results.

```python
def analyze_performance(image_path, label):
    image = cv2.imread(image_path)

    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    start_time = time.time()
    results = model(image)
    end_time = time.time()

    inference_time = end_time - start_time
    print(f"Inference Time for {image_path}: {inference_time:.4f} seconds")
    detected_objects = 0
    for result in results:
        boxes = result.boxes
        detected_objects += len(boxes)

        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].numpy()
            conf = box.conf[0].item()
            class_id = int(box.cls[0].item())

            if conf > 0.5:
                cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(image, f"Class: {class_id}, Conf: {conf:.2f}",
                            (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    print(f"Number of Detected Objects in {image_path}: {detected_objects}")

    cv2.putText(image, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.title(f'{label} Detection Results')
    plt.show()
```

This function performs object detection on an image, measures how long the detection takes, and displays the result with bounding boxes and labels. It first reads the image, checks if it loaded successfully, and then records the start and end times of the detection to calculate the inference time. For each detected object with a confidence above 0.5, it draws a bounding box and adds a label with the object’s class ID and confidence score. Finally, it overlays a custom label and displays the image with detections using Matplotlib.

```python
for img_path, label in zip(image_paths, labels):
    analyze_performance(img_path, label)
```

This loop iterates over pairs of image paths and labels, calling the analyze_performance function for each pair. For each image, it performs object detection, displays the labeled image with results, and prints the detection performance details.

*Results*

0: 512x640 7 cats, 207.3ms
Speed: 4.5ms preprocess, 207.3ms inference, 1.3ms postprocess per image at shape (1, 3, 512, 640)
Inference Time for cats1.jpg: 0.2215 seconds
Number of Detected Objects in cats1.jpg: 7

![image](https://github.com/user-attachments/assets/b9057058-3861-4031-b9ce-bdc1301b283b)

0: 512x640 1 bird, 6 cats, 2 dogs, 11 sheeps, 303.8ms
Speed: 5.0ms preprocess, 303.8ms inference, 1.7ms postprocess per image at shape (1, 3, 512, 640)
Inference Time for cats2.jpg: 0.3197 seconds
Number of Detected Objects in cats2.jpg: 20

![image](https://github.com/user-attachments/assets/b9971b61-3aa2-4568-81bc-ed68d86ad03f)

0: 512x640 2 birds, 10 cats, 3 dogs, 9 bears, 2 chairs, 303.6ms
Speed: 3.7ms preprocess, 303.6ms inference, 1.8ms postprocess per image at shape (1, 3, 512, 640)
Inference Time for cats3.jpg: 0.3168 seconds
Number of Detected Objects in cats3.jpg: 26

![image](https://github.com/user-attachments/assets/56fac0bd-dafc-4e15-b55d-c76c5424d5fe)

***Links used for information:***

* [YOLO Algorithm: Real-Time Object Detection from A to Z](https://kili-technology.com/data-labeling/machine-learning/yolo-algorithm-real-time-object-detection-from-a-to-z)
