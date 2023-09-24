---
layout: writing
title: Style Transfer
category: AI/ML
icon: /assets/projects/style-transfer/icon.jpg
tags: style-transfer art painting filters deep-learning
comment: true
urls:
  github: https://github.com/tarun-bisht/fast-style-transfer
  youtube: http://www.youtube.com/watch?v=GrS4rWifdko
---

Convert photos and videos to artwork

Using this we can stylize any photo or video in style of famous paintings using Neural Style Transfer.

### About

Neural Style Transfer was first published in the paper "A Neural Algorithm of Artistic Style" by Gatys et al., originally released in 2015. It is an image transformation technique which modifies one image in the style of another image. We take two images of content image and style image, using these two images we generate a third image which has contents from the content image while styling (textures) from style image. If we take any painting as a style image then output generated image has contents painted like style image.

This project implements two style transfer techniques, one is proposed by [Gatys et al](https://arxiv.org/abs/1508.06576) which introduce style transfer in 2015 and other was proposed by Justin Johnson in his paper [Perceptual Losses for Real-Time Style Transfer and Super-Resolution](https://arxiv.org/abs/1603.08155) which uses autoencoder netowrk to map input image to style image using same idea described in above paper, the advantage is that now if we have trained an autoencoder for one style we can use it to style multiple images efficiently without optimizing input image which makes it fast and can be used to stylize videos.

I have also written posts explaining these two papers with code so to know more about its working refer to these posts.

- [Neural Style Transfer Part 1 : Introduction](https://www.tarunbisht.com/deep%20learning/2020/12/28/neural-style-transfer-part-1-introduction/)
- [Neural Style Transfer Part 2 : Fast Style Transfer](https://www.tarunbisht.com/deep%20learning/2020/12/29/neural-style-transfer-part-2-fast-style-transfer/)

### Requirements

#### System

- For inferencing or generating images any system will work. But size of output image is limited as per system. Large images needs more momory to process. GPU is not must for inferencing but having it will be advantageous.
- For training GPU is must with tensorflow-gpu and cuda installed.
- If there is no access to GPU at local but want to train new style, there is a notebook `Fast_Style_Transfer_Colab.ipynb` open it in colab and train. For saving model checkpoints google drive is used. You can trust this notebook but I do not take any responsibility for data loss from google drive. Before running check the model save checkpoints path as it can override existing data with same name.
- Training takes around 6 hours in colab for 2 epochs.

#### Packages

- `tensorflow-gpu>=2.0` or `tensorflow>=2.0`
- `numpy`
- `matplotlib`
- `pillow`
- `opencv-python`

> This implementation is tested with tensorflow-gpu 2.0 and tensorflow-gpu 2.2 in Windows 10 and Linux

### Get Started

- Install Python3 or anaconda and install them. For detailed steps follow installation guide for [Python3](https://realpython.com/installing-python/) and [Anaconda](https://docs.anaconda.com/anaconda/install/)
- Install above packages via pip or conda. For detailed steps follow guide for [pip](https://docs.python.org/3/installing/index.html) and [conda](https://docs.anaconda.com/anaconda/user-guide/tasks/install-packages/)
- Download some [Pretrained Models](https://www.dropbox.com/sh/dkmy123bxk7f1s0/AAA-opMlprMhssPJCR1I1k4Qa?dl=0) trained on different paintings styles to start playing without need to train network
- copy and unzip checkpoints inside `data/models`
- run scripts for image and video stylization

#### Additional guides:

If stuck on Get Started Step 1 and Step 2 follow these additional resources

- [Python Installation](https://www.youtube.com/watch?v=YYXdXT2l-Gg&list)
- [pip and usage](https://www.youtube.com/watch?v=U2ZN104hIcc)
- [Anaconda installation and using conda](https://www.youtube.com/watch?v=YJC6ldI3hWk)

### Usage Instructions

- Download Github repository
- Follow [README guide](https://github.com/tarun-bisht/fast-style-transfer#how-to-use) for using the application.

### Results

![style transfer jack sparrow](https://github.com/tarun-bisht/fast-style-transfer/raw/master/output/js_candy.jpg)

![style transfer kido inazuma eleven](https://github.com/tarun-bisht/fast-style-transfer/raw/master/output/kido.jpg)

![style transfer webcam](https://github.com/tarun-bisht/fast-style-transfer/raw/master/output/webcam.gif)

![style transfer video output](https://github.com/tarun-bisht/fast-style-transfer/raw/master/output/video.gif)
