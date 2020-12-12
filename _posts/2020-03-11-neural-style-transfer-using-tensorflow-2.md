---
layout: writing
title:  neural style transfer using tensorflow 2
date:   2020-03-11 23:03:50 +0530
category: Deep Learning
tags: style-image python tensorflow2 intermediate
---
When the style of one image is mixed with content of another image then it is called Style Transfer and we are using a neural network to do so call Neural Style Transfer. As we are dealing with images so we need a convolutional neural network.
Neural Style Transfer was first published in the paper “A Neural Algorithm of Artistic Style” by Gatys et al.,
 originally released in 2015 and in this session, we are implementing this paper using Tensorflow 2.0.
 
 <!-- more -->

[arXiv paper link](https://arxiv.org/abs/1508.06576)

![Messi Stylized Image](https://cdn-images-1.medium.com/1*RE-1XL15oejnCSbgyj11vg.png)

The convolutional neural network we will use is the VGG19 model. VGG19 is a convolutional neural network that is trained on more than a million images from the ImageNet database. The network is 19 layers deep and can classify images into 1000 object categories, such as a keyboard, mouse, pencil, and many animals.

This model is suggested by the paper and it gives good results. Also, we will use pre-trained weights for models on the ImageNet dataset so that it’s hidden layers (deep layers) have learned of images i.e. When we give the model an image as input its hidden layer can build representations of the image. Lower layers in the network represent the low-level feature of the image while a higher layer in the network represents a complex and high-level features.

In short, we are actually making an image of how convolutional networks are representing an image.

Lower layers of the network will capture styles (low-level features) and higher layers will capture the content of the image (high-level features)

### Some Important points

- Model we use is VGG19 with pre-trained weights and with no dense layers (these dense layers in model are used for classification so we do not need it since we are not dealing with classification).

- We are using image representations for both style and content image by the network so make sure you understand convolutional neural networks and how its hidden convolutional layer represents a feature of the image.

- Generated image= The image which the network will generate or output.

For neural style transfer, we define a loss function and minimize it to generate a resultant image.

Loss Function to minimize neural style transfer is given by

![Loss Function](https://cdn-images-1.medium.com/max/800/1*JsqfM5hNn3cL9IJbMZR5kg.png)

### Steps to tackle

- Initialize generated image (G) with random values.

- Use gradient descent or variant to minimize J(G)

- Update generated image values by applying gradient calculated in the above step

![updating generated image](https://cdn-images-1.medium.com/max/800/1*SKWcdg1TWFTPMgAOKOfsJg.png)


### Content Cost Function

- Let L be hidden layer to compute the content cost

- Let a[L][C] and a[L][G] be activation of layer L for the image.

- If both a[L][C] and a[L][G] are similar both images have the same content.

![Content Loss](https://cdn-images-1.medium.com/max/800/1*9AcWLV_XJ8obhBl4ZI2-tQ.png)


### Style Cost Function

- Let we are using layer L activation to measure style.

- Define style as the correlation between activation across channels

- Let a<sup>[l]</sup><sub>ijk</sub>= activation at (i,j,k) where i=height, j=width, k=channel

- Then we define the style matrix (gram matrix) denotes the correlation between channel K and K’

![Gram Matrix](https://cdn-images-1.medium.com/max/800/1*aJSyMZzcrk16fBfXVRAQpQ.png)

- In a more intuitive way, gram matrix can be seen as how similar two images are similar, Its dot product between two vectors of activation at layer L the lesser the angle between them or more closer the respective coordinates. So the more similar they are, the larger the dot product gets.

![Style Loss](https://cdn-images-1.medium.com/max/800/1*Sr0aXBBTXkBw3pvx7bDfQQ.png)


Now we have defined Content Loss and Style loss putting these in Total Loss equation and we get the total loss.

### Implementation

Open [https://colab.research.google.com](https://colab.research.google.com/) or an IPython notebook and start.

Please Consider opening first to understand better

[**Google Colaboratory**  
colab.research.google.com](https://colab.research.google.com/drive/17D7J_ScGuIYh966Q0wfXqVcSX1J57VAk "https://colab.research.google.com/drive/17D7J_ScGuIYh966Q0wfXqVcSX1J57VAk")

or

**[tarun-bisht/tensorflow-scripts](https://github.com/tarun-bisht/tensorflow-scripts/tree/master/Neural%20Style%20Transfer)**

I have commented on every section to understand well. I am only posting some screenshots of the same notebook.

**Installing Tensorflow 2.0**

![Installing Tensorflow](https://miro.medium.com/max/366/1*XB5EXKMQ_LkopYQaUcw97Q.png)


**Importing Dependencies**

![Importing Dependencies](https://miro.medium.com/max/537/1*j145QW66Sx48m7yofleFlg.png)

**Helper Functions**

![Load Image from URL](https://miro.medium.com/max/452/1*76m4z4A-w_W4aRZNbWmiXQ.png)

![Plot Image to grid](https://miro.medium.com/max/577/1*sDoZqHIBcoGdQwyW6lxZyQ.png)

![plotting graph](https://miro.medium.com/max/289/1*UiNrhSxrLPIFCLCUN1PyVg.png)

**Defining Layers from which we will get activations**

Lower layers of a convolutional extract low-level feature from images and its progress to learn high-level features from images as we progress to a high level of the network. So we have used low layers for extracting style and color and take high-level layer for extracting the content of image.

![layers activation](https://miro.medium.com/max/1143/1*cjRS6i_ti2BQSN9cgnrdTg.png)

**Creating Model using Keras Functional API**

Pooling is set to average pooling as suggested by the research paper of neural style transfer. Model is created by providing it input and output layers. The input layer is the default for VGG but output layer are now our content and style layers we defined.

![creating model](https://miro.medium.com/max/1260/1*NIgIayPZtYVO1fF0sN8Hbw.png)

**Preprocess Image to be sent as input in VGG Model**

VGG model takes images as BGR format instead of RGB format so it needed to be preprocessed first

![processed content and style image](https://miro.medium.com/max/727/1*g0VVbIibOu4CIhnPqmtcLw.png)

![Helper Function to Reprocess the preprocessed image](https://miro.medium.com/max/1321/1*vrCoDzOHm3axvD11wtqYGA.png)

**The Loss Function for Neural Style Transfer**

![Content Loss](https://miro.medium.com/max/1229/1*hgN2V54E8Qty-PXCNW84hQ.png)

![Gram Matrix and Style Loss](https://miro.medium.com/max/1283/1*1vaXbZd3L7Sh97C87efBKg.png)


![Total Loss](https://miro.medium.com/max/1217/1*hWnn1pJEaAn0oF1Ko7Q3vw.png)


**Optimizing Loss**

Optimization Function optimize generated image values based on minimizing the total loss function

![Optimizing Loss and generating images](https://miro.medium.com/max/1278/1*ZzYdGOtkH_ZGX-S2w9yj0Q.png)


So, I Hope, this article is helpful for someone understanding and implementing neural style transfer. Thanks for being till last.

### More Results

![style1 image](https://miro.medium.com/max/601/1*sw3n6ndsqx0WCqbL9oDnHQ.png)

![style2 image](https://miro.medium.com/max/601/1*yU_UJ4PkIEz-uAJGwVBwRw.png)

### Full Code

[**tarun-bisht/tensorflow-scripts**](https://github.com/tarun-bisht/tensorflow-scripts/tree/master/Neural%20Style%20Transfer "https://github.com/tarun-bisht/tensorflow-scripts/tree/master/Neural%20Style%20Transfer")

[**Google Colaboratory**](https://colab.research.google.com/drive/17D7J_ScGuIYh966Q0wfXqVcSX1J57VAk "https://colab.research.google.com/drive/17D7J_ScGuIYh966Q0wfXqVcSX1J57VAk")

[**Medium Article Link**](https://medium.com/@tarunbishttarun11/neural-style-transfer-in-tensorflow-2-0-2235cd3f6b8b)