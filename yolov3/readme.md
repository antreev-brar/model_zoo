# Tensorflow Implementation of YOLOv3

Download weights in the repo containing files
```bash 
$ wget https://pjreddie.com/media/files/yolov3.weights
```
## Usage
```bash
$ python3 main.py 
```
> **_NOTE:_** on Colab Notebook use following command:
```python
!git clone https://github.com/antreev-brar/model-zoo.git
%cd /content/model-zoo/yolov3
!wget https://pjreddie.com/media/files/yolov3.weights
!python main.py
```
## Contributed by:
* [Antreev Singh Brar](https://github.com/antreev-brar)
## References

* **Title**: YOLOv3: An Incremental Improvement
* **Authors**: Joseph Redmon, Ali Farhadi
* **Link**: https://arxiv.org/abs/1804.02767v1
* **Tags**: Neural Network
* **Year**: 2018

# Summary

## Why new Model(Drawbacks of YOLOv1 and YOLOv3)

* YOLOv1 imposes strong spatial constraints on bounding box predictions since each grid cell only predicts two boxes and can only have one class. This spatial constraint limits the number of nearby objects that our model can predict. Model struggles with small objects that appear in groups, such as flocks of birds. Since the model learns to predict bounding boxes from data, it struggles to generalize to objects in new or unusual aspect ratios or configurations.  Model also uses relatively coarse features for predicting bounding boxes since their architecture has multiple downsampling layers from the input image. Finally, while  train on a loss function that approximates detection performance, their loss function treats errors the same in small bounding boxes versus large bounding boxes. A small error in a large box is generally benign but a small error in a small box has a much greater effect on IOU. The main source of error is incorrect localizations
* YOLO v2 used a custom deep architecture darknet-19, an originally 19-layer network supplemented with 11 more layers for object detection. With a 30-layer architecture, YOLO v2 often struggled with small object detections. This was attributed to loss of fine-grained features as the layers downsampled the input.YOLO v2’s architecture was still lacking some of the most important elements that are now staple in most of state-of-the art algorithms. No residual blocks, no skip connections and no upsampling. YOLO v2 used mean squared error for loss function which isn't the most suitable choice .
## Introduction 
Main purpose of a object detector is to be fast and accurate and able to recognize wide dataset.So they suggest a new model with some features modified as well as added to fulfill its purpose

## Accuracy improvement
### 1-Batch normalization
   Add batch normalization in convolution layers. This removes the need for dropouts and pushes mAP up 2%.
### 2-Convolutional with Anchor Boxes

* As indicated in the YOLO paper, the early training is susceptible to unstable gradients. Initially, YOLO makes arbitrary guesses on the boundary boxes. These guesses may work well for some objects but badly for others resulting in steep gradient changes. In early training, predictions are fighting with each other on what shapes to specialize on.
* YOLO predicts the coordinates of bounding boxes directly using fully connected layers on top of the convolutional feature extractor. Predicting offsets instead of coordinates simplifies the problem and makes it easier for the network to learn. We remove the fully connected layers from YOLO and use anchor boxes to predict bounding boxes. Using anchor boxes we get a small decrease in accuracy.
* Instead of choosing priors by hand, we run k-means clustering on the training set bounding boxes to automatically find good priors. If we use standard k-means with Euclidean distance learger boxes generate more error than smaller boxes. However, what we really want are priros that lead to good IOU scores, which is indepedndent of the size of the box. Thus for our distance metric we use 1 - IOU(box,centroid). This is how they chooses anchor box...
![4](./assets/anchor.png)
### 3-High-resolution classifier
The YOLO training composes of 2 phases. First, we train a classifier network like VGG16. Then we replace the fully connected layers with a convolution layer and retrain it end-to-end for the object detection. YOLO trains the classifier with 224 × 224 pictures followed by 448 × 448 pictures for the object detection. YOLOv2 starts with 224 × 224 pictures for the classifier training but then retune the classifier again with 448 × 448 pictures using much fewer epochs. This makes the detector training easier and moves mAP up by 4%.

### 4-Capability improvement

They suggest a method to predict bounding boxes of the 9000 most common classes in ImageNet. They add a few more abstract classes to that (e.g. dog for all breeds of dogs) and arrive at over 9000 classes (9418 to be precise).
They train on ImageNet and MSCOCO.
![4](./assets/mix2.png)
### 5-Direct location prediction
* YOLOv1 does not have constraints on location prediction which makes the model unstable at early iterations. The predicted bounding box can be far from the original grid location.
* YOLOv2 bounds the location using logistic activation σ, which makes the value fall between 0 to 1:
![4](./assets/bounding.png)
### Model Summary
```
__________________________________________________________________________________________________

Total params: 62,001,757
Trainable params: 61,949,149
Non-trainable params: 52,608
```
I used yolov3 pretrained on MSCOCO dataset

# Results

## Images after 50 epoch(VOC 2012)

![4](./assets/result2.png)
![4](./assets/result3.png)
![4](./assets/result.png)

## Accuracy and speed of Model(VOC 2007)

![4](./assets/acc.png)
