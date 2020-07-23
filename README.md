# Model-zoo
This repo contains my Summer Project " Model-zoo", under the science and technology council, IITk 


Deep learning is an artificial intelligence (AI) function that imitates the workings of the human brain in processing data and creating patterns for use in decision making. Deep learning is a subset of machine learning in artificial intelligence that has networks capable of learning unsupervised from data that is unstructured or unlabeled. Also known as deep neural learning or deep neural network.

### Overview

I implemented 4 models spanning various categories using TensorFlow which is one of the most popular Deep Learning frameworks for implementation of the models.Each model is in a separate directory with a README including the usage and architecture details.

- **Resnet** The ”levels” of features in a Deep-CNN can be enriched with the number of stacked
layers(depth). The problem lies in the fact that models that are too deep have a hard time
optimizing their parameters. These problems are addressed by introducing a Deep Residual
learning framework, in which the layers are made to fit a residual mapping, which is easier to
optimize.

- **Image Captioning** Earlier models rely on some hard-coded visual concepts and sentence
templates, limiting the scope of model. Deep Visual-Semantic Alignments for Generating Image
Descriptions model uses a Deep Neural Network (DNN) and a multimodal Recurrent Neural
Network (RNN) architecture to take an input image and generates its description in text. This
implementation would be a great help in CCTV cameras, blind aid or even search engines.

- **YOLO v3** It uses an even deeper Darknet-53 network ditching softmax as it is not the most
suitable choice. Instead, Independent logistic classifiers are used and binary cross-entropy loss
is used and bounding boxes are predicted on 3 different scales for detection on different scales

- **SS-GAN** The semi-supervised GAN, or SGAN, model is an extension of the GAN architecture that involves the simultaneous training of a supervised discriminator, unsupervised discriminator, and a generator model. The result is both a supervised classification model that generalizes well to unseen examples and a generator model that outputs plausible examples of images from the domain.
