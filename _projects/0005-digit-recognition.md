---
layout: writing
title: Digit Recognition
category: Web Apps
icon: /assets/projects/digit-recognition/icon.png
tags: digit-recognition mnist
comment: true
urls:
  github: https://github.com/tarun-bisht/digit-recognition
  live: http://tarun-bisht.github.io/digit-recognition/
---

- A Web Application developed with Tensorflowjs recognize handwritten digits(0-9).
- It is done using Convolutional Neural Network.
- User will have to write a digit in pad provided and it will inference what digit was written.
- This project provide basic structure for creating machine learning based web applications.
- The model used here was trained in python with tensorflow then model was converted to tensorflowjs model. This trained model is hosted in firebase hosting [Here](https://models-lib.web.app/models/mnist_digits/model.json).
- Inferencing from model is done in client side using javascript with tensorflowjs.

## Resources used

- [Kaggle Notebook](https://www.kaggle.com/tarunbisht11/mnist-digit-recognition-convnet-with-leakyrelu)
- [Hosted Model](https://models-lib.web.app/models/mnist_digits/model.json)
- [Drawing Pad Used in Project](https://github.com/tarun-bisht/SimpleDrawingPad)
