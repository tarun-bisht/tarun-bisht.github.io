---
layout: writing
title: Security Camera
category: AI/ML Course-Projects Research-Projects
icon: /assets/projects/security-camera/icon.jpg
tags: object-detection camera surveillance cctv
comment: true
urls:
  github: https://github.com/tarun-bisht/security-camera
---

Intelligent Security camera application powered by AI. It uses live video stream from camera feed or rtsp streaming from ip camera or cctv and use object detection to detect intruders in these feeds, when detected it send alert into mail along with the image it sensed as intruder so user can verify on spot. It also start recording video from the point it detects an intruder till that intruder is on sight of camera. This project is part of my work on a research project that deals with monkey theft detection and alert user when detected.

[Here](/assets/projects/security-camera/application-of-object-detection-in-home-surveillance-system.pdf) is very detailed writeup of this project that was part of my master's thesis.

## Requirements

### System

System requirements depends on complexity or size of object detection model, larger model will require more compute power and will be good at detection. I have used this in a raspberry pi 3b with a pi camera using mobilenet backbone and it gave around 1-2 fps. For final implementation tflite model was used with pi which boost fps of application. With my Nvidia Geforce 940 MX GPU based system it is giving around 20-30 fps with mobilenet backbone. So here is tradeoff between compute power and accuracy.

### Python 3

Python 3.6 or higher. Tested with Python 3.6, 3.7, 3.8, 3.9 in Windows 10 and Linux.

### Packages

- `tensorflow-gpu>=2.0` or `tensorflow>=2.0`
- `numpy`
- `absl-py`
- `opencv-python`

This implementation is tested with tensorflow cpu and gpu 2.0, 2.2, 2.7 in Windows 10 and Linux.

## Detection Results

<div>
  <a href="https://youtu.be/FK4kqej6t5Q"><img src='https://raw.githubusercontent.com/tarun-bisht/security-camera/master/data/outputs/monkey_detection.gif' alt="Monkey Detection Object Detection"></a>
</div>

## Other Links

- [Training a detection model for camera](https://github.com/tarun-bisht/tensorflow-object-detection)
- [TFlite based security camera for low compute devices](https://github.com/tarun-bisht/security-camera-tflite)
