---
layout: writing
title:  deepdream using tensorflow an introduction to generative deep learning
date:   2020-05-05 22:31:50 +0530
category: Deep Learning
tags: python intermediate tensorflow inception
---
DeepDream is image modification algorithm an example of generative deep learning that uses representation learned by convolution neural networks to modify images. It was released by Google in 2015. The popularity of deepdream caused due to its crappy artifacts in images, from eyes to feathers to dog faces. It was initially created to help scientists and engineers to see what a deep neural network is seeing when it is looks given input. 
<!-- more -->

DeepDream is based on one of the techniques of visualizing learnings of convnets. Using that technique we can visualize patterns that activate a given layer of convolutional neural network or visual pattern that each filter respond to in convolutional layers. This is done by applying gradient ascent in input space, which maximizes the response of the specific filter in convnets.

### Gradient Ascent

gradient ascent is opposite of gradient descent. Both are optimization algorithms. As gradient descent finds out minima of a function gradient ascent finds out maxima of a function. The process of gradient ascent is same as gradient descent we first find out gradient(derivative) of function with respect to our training parameters and then change training parameters so as to maximize instead of minimizing by moving it in opposite direction of gradient descent. 

For visualizing patterns learned by convnets we have to maximize the response of specific filters. In simple words, we have a response(activations) of the specific filter in a convolutional layer and we change our input space to maximize that filter's response by using gradient ascent.

### Steps to create Deepdream

- we start with an image and pass it to a pretrained convolutional neural network like inception or vgg
- we try to maximize activation of entire layer rather than specific filter for this we define a simple loss function which will maximize activations of layers on maximizing that loss function. So we use mean of activations of layers as loss function.
- finally we will change our input space(image) by applying that gradient to image which eventually will maximize out loss function.
- additional steps like tiling and octaves are needed in order to work with large images so that it can be fit efficiently
on RAM and provide better results.

Implementing deepdream teaches a lot of other concepts of deep learning. It breaks the rule of traditional *model.fit* in every deep learning problem. Also playing with result is quite interesting so get ready for deepdream.


{% highlight python linenos %}
input_img_path="starry.jpg"
{% endhighlight %}

Starting with our input image in which deepdream patterns will be shown as output. First we will define path of our input image.


{% highlight python linenos %}
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import inception_v3
from tensorflow.keras.models import load_model,Model
from PIL import Image
import IPython.display as display
import time
{% endhighlight %}

Next we will import all dependencies which we will need for creating deepdream. 

- **numpy :** for arrays manipulation
- **tensorflow :** for tensor operations
- **tensorflow.keras :** high level neural network library for tensorflow for creating neural networks
- **pillow :** for converting an image to numpy array and numpy array to image, saving out output image.
- **Ipython.display :** for displaying images in notebook
- **time :** for calculating time of each iteration

We are using Inception pretrained model for this as it produces better outputs of deepdreams and original implementation also used Inception model.

> Dreams in Inception movie.😁😁


{% highlight python linenos %}
def load_image(image_path,max_dim=512):
    img=Image.open(image_path)
    img=img.convert("RGB")
    img.thumbnail([max_dim,max_dim])
    img=np.array(img,dtype=np.uint8)
    img=np.expand_dims(img,axis=0)
    return img
{% endhighlight %}

the above function 
- loads image from path
- convert it into RGB format
- resize it with max dimension specified while maintaining aspect ratio
- converting an image to numpy array and creating a batch of a single image since neural networks expects the input to be in batches. 


{% highlight python linenos %}
def deprocess_inception_image(img):
    img = 255*(img+1.0)/2.0
    return np.array(img, np.uint8)
{% endhighlight %}

the above function cancels out effects of preprocessing applied by inception's preprocess_input function. preprocess_input function for inception model scales down pixels of image to be in range -1 to 1 so this function will scale pixels to be in range 0 to 255


{% highlight python linenos %}
def array_to_img(array,deprocessing=False):
    if deprocessing:
        array=deprocess_inception_image(array)
    if np.ndim(array)>3:
        assert array.shape[0]==1
        array=array[0]
    return Image.fromarray(array)
{% endhighlight %}

the above function will convert array to image. if deprocessing is true it will first deprocess inception preprocessing and then convert array to image


{% highlight python linenos %}
def show_image(img):
    image=array_to_img(img)
    display.display(image)
{% endhighlight %}

the above function will show image in notebook by first converting array to image


{% highlight python linenos %}
input_image=load_image(input_img_path,max_dim=512)
print(input_image.shape)
show_image(input_image)
{% endhighlight %}

    (1, 338, 512, 3)
    


![png](https://storage.googleapis.com/tarun-bisht.appspot.com/blogs/deep_dream_10ec0c4529c1aafed)


Now lets load our input image and display it.


{% highlight python linenos %}
preprocessed_image=inception_v3.preprocess_input(input_image)
show_image(deprocess_inception_image(preprocessed_image))
{% endhighlight %}


![png](https://storage.googleapis.com/tarun-bisht.appspot.com/blogs/deep_dream_29905fe4cd8a5744b)


Also check if our deprocess_image function is working as expected


{% highlight python linenos %}
def deep_dream_model(model,layer_names):
    model.trainable=False
    outputs=[model.get_layer(name).output for name in layer_names]
    new_model=Model(inputs=model.input,outputs=outputs)
    return new_model
{% endhighlight %}

the above function creates a deepdream model. Since we are not training our model so set trainable to false. Our deepdream model takes input as image and outputs the activations of layers which we will use to embed patterns learned by that layers into our input image


{% highlight python linenos %}
inception=inception_v3.InceptionV3(weights="imagenet",include_top=False)
inception.summary()
{% endhighlight %}

    Model: "inception_v3"
    __________________________________________________________________________________________________
    Layer (type)                    Output Shape         Param #     Connected to                     
    ==================================================================================================
    input_1 (InputLayer)            [(None, None, None,  0                                            
    __________________________________________________________________________________________________
    conv2d (Conv2D)                 (None, None, None, 3 864         input_1[0][0]                    
    __________________________________________________________________________________________________
    batch_normalization (BatchNorma (None, None, None, 3 96          conv2d[0][0]                     
    __________________________________________________________________________________________________
    activation (Activation)         (None, None, None, 3 0           batch_normalization[0][0]        
    __________________________________________________________________________________________________
    conv2d_1 (Conv2D)               (None, None, None, 3 9216        activation[0][0]                 
    __________________________________________________________________________________________________
    batch_normalization_1 (BatchNor (None, None, None, 3 96          conv2d_1[0][0]                   
    __________________________________________________________________________________________________
    activation_1 (Activation)       (None, None, None, 3 0           batch_normalization_1[0][0]      
    __________________________________________________________________________________________________
    conv2d_2 (Conv2D)               (None, None, None, 6 18432       activation_1[0][0]               
    __________________________________________________________________________________________________
    batch_normalization_2 (BatchNor (None, None, None, 6 192         conv2d_2[0][0]                   
    __________________________________________________________________________________________________
    activation_2 (Activation)       (None, None, None, 6 0           batch_normalization_2[0][0]      
    __________________________________________________________________________________________________
    max_pooling2d (MaxPooling2D)    (None, None, None, 6 0           activation_2[0][0]               
    __________________________________________________________________________________________________
    conv2d_3 (Conv2D)               (None, None, None, 8 5120        max_pooling2d[0][0]              
    __________________________________________________________________________________________________
    batch_normalization_3 (BatchNor (None, None, None, 8 240         conv2d_3[0][0]                   
    __________________________________________________________________________________________________
    activation_3 (Activation)       (None, None, None, 8 0           batch_normalization_3[0][0]      
    __________________________________________________________________________________________________
    conv2d_4 (Conv2D)               (None, None, None, 1 138240      activation_3[0][0]               
    __________________________________________________________________________________________________
    batch_normalization_4 (BatchNor (None, None, None, 1 576         conv2d_4[0][0]                   
    __________________________________________________________________________________________________
    activation_4 (Activation)       (None, None, None, 1 0           batch_normalization_4[0][0]      
    __________________________________________________________________________________________________
    max_pooling2d_1 (MaxPooling2D)  (None, None, None, 1 0           activation_4[0][0]               
    __________________________________________________________________________________________________
    conv2d_8 (Conv2D)               (None, None, None, 6 12288       max_pooling2d_1[0][0]            
    __________________________________________________________________________________________________
    batch_normalization_8 (BatchNor (None, None, None, 6 192         conv2d_8[0][0]                   
    __________________________________________________________________________________________________
    activation_8 (Activation)       (None, None, None, 6 0           batch_normalization_8[0][0]      
    __________________________________________________________________________________________________
    conv2d_6 (Conv2D)               (None, None, None, 4 9216        max_pooling2d_1[0][0]            
    __________________________________________________________________________________________________
    conv2d_9 (Conv2D)               (None, None, None, 9 55296       activation_8[0][0]               
    __________________________________________________________________________________________________
    batch_normalization_6 (BatchNor (None, None, None, 4 144         conv2d_6[0][0]                   
    __________________________________________________________________________________________________
    batch_normalization_9 (BatchNor (None, None, None, 9 288         conv2d_9[0][0]                   
    __________________________________________________________________________________________________
    activation_6 (Activation)       (None, None, None, 4 0           batch_normalization_6[0][0]      
    __________________________________________________________________________________________________
    activation_9 (Activation)       (None, None, None, 9 0           batch_normalization_9[0][0]      
    __________________________________________________________________________________________________
    average_pooling2d (AveragePooli (None, None, None, 1 0           max_pooling2d_1[0][0]            
    __________________________________________________________________________________________________
    conv2d_5 (Conv2D)               (None, None, None, 6 12288       max_pooling2d_1[0][0]            
    __________________________________________________________________________________________________
    conv2d_7 (Conv2D)               (None, None, None, 6 76800       activation_6[0][0]               
    __________________________________________________________________________________________________
    conv2d_10 (Conv2D)              (None, None, None, 9 82944       activation_9[0][0]               
    __________________________________________________________________________________________________
    conv2d_11 (Conv2D)              (None, None, None, 3 6144        average_pooling2d[0][0]          
    __________________________________________________________________________________________________
    batch_normalization_5 (BatchNor (None, None, None, 6 192         conv2d_5[0][0]                   
    __________________________________________________________________________________________________
    batch_normalization_7 (BatchNor (None, None, None, 6 192         conv2d_7[0][0]                   
    __________________________________________________________________________________________________
    batch_normalization_10 (BatchNo (None, None, None, 9 288         conv2d_10[0][0]                  
    __________________________________________________________________________________________________
    batch_normalization_11 (BatchNo (None, None, None, 3 96          conv2d_11[0][0]                  
    __________________________________________________________________________________________________
    activation_5 (Activation)       (None, None, None, 6 0           batch_normalization_5[0][0]      
    __________________________________________________________________________________________________
    activation_7 (Activation)       (None, None, None, 6 0           batch_normalization_7[0][0]      
    __________________________________________________________________________________________________
    activation_10 (Activation)      (None, None, None, 9 0           batch_normalization_10[0][0]     
    __________________________________________________________________________________________________
    activation_11 (Activation)      (None, None, None, 3 0           batch_normalization_11[0][0]     
    __________________________________________________________________________________________________
    mixed0 (Concatenate)            (None, None, None, 2 0           activation_5[0][0]               
                                                                     activation_7[0][0]               
                                                                     activation_10[0][0]              
                                                                     activation_11[0][0]              
    __________________________________________________________________________________________________
    conv2d_15 (Conv2D)              (None, None, None, 6 16384       mixed0[0][0]                     
    __________________________________________________________________________________________________
    batch_normalization_15 (BatchNo (None, None, None, 6 192         conv2d_15[0][0]                  
    __________________________________________________________________________________________________
    activation_15 (Activation)      (None, None, None, 6 0           batch_normalization_15[0][0]     
    __________________________________________________________________________________________________
    conv2d_13 (Conv2D)              (None, None, None, 4 12288       mixed0[0][0]                     
    __________________________________________________________________________________________________
    conv2d_16 (Conv2D)              (None, None, None, 9 55296       activation_15[0][0]              
    __________________________________________________________________________________________________
    batch_normalization_13 (BatchNo (None, None, None, 4 144         conv2d_13[0][0]                  
    __________________________________________________________________________________________________
    batch_normalization_16 (BatchNo (None, None, None, 9 288         conv2d_16[0][0]                  
    __________________________________________________________________________________________________
    activation_13 (Activation)      (None, None, None, 4 0           batch_normalization_13[0][0]     
    __________________________________________________________________________________________________
    activation_16 (Activation)      (None, None, None, 9 0           batch_normalization_16[0][0]     
    __________________________________________________________________________________________________
    average_pooling2d_1 (AveragePoo (None, None, None, 2 0           mixed0[0][0]                     
    __________________________________________________________________________________________________
    conv2d_12 (Conv2D)              (None, None, None, 6 16384       mixed0[0][0]                     
    __________________________________________________________________________________________________
    conv2d_14 (Conv2D)              (None, None, None, 6 76800       activation_13[0][0]              
    __________________________________________________________________________________________________
    conv2d_17 (Conv2D)              (None, None, None, 9 82944       activation_16[0][0]              
    __________________________________________________________________________________________________
    conv2d_18 (Conv2D)              (None, None, None, 6 16384       average_pooling2d_1[0][0]        
    __________________________________________________________________________________________________
    batch_normalization_12 (BatchNo (None, None, None, 6 192         conv2d_12[0][0]                  
    __________________________________________________________________________________________________
    batch_normalization_14 (BatchNo (None, None, None, 6 192         conv2d_14[0][0]                  
    __________________________________________________________________________________________________
    batch_normalization_17 (BatchNo (None, None, None, 9 288         conv2d_17[0][0]                  
    __________________________________________________________________________________________________
    batch_normalization_18 (BatchNo (None, None, None, 6 192         conv2d_18[0][0]                  
    __________________________________________________________________________________________________
    activation_12 (Activation)      (None, None, None, 6 0           batch_normalization_12[0][0]     
    __________________________________________________________________________________________________
    activation_14 (Activation)      (None, None, None, 6 0           batch_normalization_14[0][0]     
    __________________________________________________________________________________________________
    activation_17 (Activation)      (None, None, None, 9 0           batch_normalization_17[0][0]     
    __________________________________________________________________________________________________
    activation_18 (Activation)      (None, None, None, 6 0           batch_normalization_18[0][0]     
    __________________________________________________________________________________________________
    mixed1 (Concatenate)            (None, None, None, 2 0           activation_12[0][0]              
                                                                     activation_14[0][0]              
                                                                     activation_17[0][0]              
                                                                     activation_18[0][0]              
    __________________________________________________________________________________________________
    conv2d_22 (Conv2D)              (None, None, None, 6 18432       mixed1[0][0]                     
    __________________________________________________________________________________________________
    batch_normalization_22 (BatchNo (None, None, None, 6 192         conv2d_22[0][0]                  
    __________________________________________________________________________________________________
    activation_22 (Activation)      (None, None, None, 6 0           batch_normalization_22[0][0]     
    __________________________________________________________________________________________________
    conv2d_20 (Conv2D)              (None, None, None, 4 13824       mixed1[0][0]                     
    __________________________________________________________________________________________________
    conv2d_23 (Conv2D)              (None, None, None, 9 55296       activation_22[0][0]              
    __________________________________________________________________________________________________
    batch_normalization_20 (BatchNo (None, None, None, 4 144         conv2d_20[0][0]                  
    __________________________________________________________________________________________________
    batch_normalization_23 (BatchNo (None, None, None, 9 288         conv2d_23[0][0]                  
    __________________________________________________________________________________________________
    activation_20 (Activation)      (None, None, None, 4 0           batch_normalization_20[0][0]     
    __________________________________________________________________________________________________
    activation_23 (Activation)      (None, None, None, 9 0           batch_normalization_23[0][0]     
    __________________________________________________________________________________________________
    average_pooling2d_2 (AveragePoo (None, None, None, 2 0           mixed1[0][0]                     
    __________________________________________________________________________________________________
    conv2d_19 (Conv2D)              (None, None, None, 6 18432       mixed1[0][0]                     
    __________________________________________________________________________________________________
    conv2d_21 (Conv2D)              (None, None, None, 6 76800       activation_20[0][0]              
    __________________________________________________________________________________________________
    conv2d_24 (Conv2D)              (None, None, None, 9 82944       activation_23[0][0]              
    __________________________________________________________________________________________________
    conv2d_25 (Conv2D)              (None, None, None, 6 18432       average_pooling2d_2[0][0]        
    __________________________________________________________________________________________________
    batch_normalization_19 (BatchNo (None, None, None, 6 192         conv2d_19[0][0]                  
    __________________________________________________________________________________________________
    batch_normalization_21 (BatchNo (None, None, None, 6 192         conv2d_21[0][0]                  
    __________________________________________________________________________________________________
    batch_normalization_24 (BatchNo (None, None, None, 9 288         conv2d_24[0][0]                  
    __________________________________________________________________________________________________
    batch_normalization_25 (BatchNo (None, None, None, 6 192         conv2d_25[0][0]                  
    __________________________________________________________________________________________________
    activation_19 (Activation)      (None, None, None, 6 0           batch_normalization_19[0][0]     
    __________________________________________________________________________________________________
    activation_21 (Activation)      (None, None, None, 6 0           batch_normalization_21[0][0]     
    __________________________________________________________________________________________________
    activation_24 (Activation)      (None, None, None, 9 0           batch_normalization_24[0][0]     
    __________________________________________________________________________________________________
    activation_25 (Activation)      (None, None, None, 6 0           batch_normalization_25[0][0]     
    __________________________________________________________________________________________________
    mixed2 (Concatenate)            (None, None, None, 2 0           activation_19[0][0]              
                                                                     activation_21[0][0]              
                                                                     activation_24[0][0]              
                                                                     activation_25[0][0]              
    __________________________________________________________________________________________________
    conv2d_27 (Conv2D)              (None, None, None, 6 18432       mixed2[0][0]                     
    __________________________________________________________________________________________________
    batch_normalization_27 (BatchNo (None, None, None, 6 192         conv2d_27[0][0]                  
    __________________________________________________________________________________________________
    activation_27 (Activation)      (None, None, None, 6 0           batch_normalization_27[0][0]     
    __________________________________________________________________________________________________
    conv2d_28 (Conv2D)              (None, None, None, 9 55296       activation_27[0][0]              
    __________________________________________________________________________________________________
    batch_normalization_28 (BatchNo (None, None, None, 9 288         conv2d_28[0][0]                  
    __________________________________________________________________________________________________
    activation_28 (Activation)      (None, None, None, 9 0           batch_normalization_28[0][0]     
    __________________________________________________________________________________________________
    conv2d_26 (Conv2D)              (None, None, None, 3 995328      mixed2[0][0]                     
    __________________________________________________________________________________________________
    conv2d_29 (Conv2D)              (None, None, None, 9 82944       activation_28[0][0]              
    __________________________________________________________________________________________________
    batch_normalization_26 (BatchNo (None, None, None, 3 1152        conv2d_26[0][0]                  
    __________________________________________________________________________________________________
    batch_normalization_29 (BatchNo (None, None, None, 9 288         conv2d_29[0][0]                  
    __________________________________________________________________________________________________
    activation_26 (Activation)      (None, None, None, 3 0           batch_normalization_26[0][0]     
    __________________________________________________________________________________________________
    activation_29 (Activation)      (None, None, None, 9 0           batch_normalization_29[0][0]     
    __________________________________________________________________________________________________
    max_pooling2d_2 (MaxPooling2D)  (None, None, None, 2 0           mixed2[0][0]                     
    __________________________________________________________________________________________________
    mixed3 (Concatenate)            (None, None, None, 7 0           activation_26[0][0]              
                                                                     activation_29[0][0]              
                                                                     max_pooling2d_2[0][0]            
    __________________________________________________________________________________________________
    conv2d_34 (Conv2D)              (None, None, None, 1 98304       mixed3[0][0]                     
    __________________________________________________________________________________________________
    batch_normalization_34 (BatchNo (None, None, None, 1 384         conv2d_34[0][0]                  
    __________________________________________________________________________________________________
    activation_34 (Activation)      (None, None, None, 1 0           batch_normalization_34[0][0]     
    __________________________________________________________________________________________________
    conv2d_35 (Conv2D)              (None, None, None, 1 114688      activation_34[0][0]              
    __________________________________________________________________________________________________
    batch_normalization_35 (BatchNo (None, None, None, 1 384         conv2d_35[0][0]                  
    __________________________________________________________________________________________________
    activation_35 (Activation)      (None, None, None, 1 0           batch_normalization_35[0][0]     
    __________________________________________________________________________________________________
    conv2d_31 (Conv2D)              (None, None, None, 1 98304       mixed3[0][0]                     
    __________________________________________________________________________________________________
    conv2d_36 (Conv2D)              (None, None, None, 1 114688      activation_35[0][0]              
    __________________________________________________________________________________________________
    batch_normalization_31 (BatchNo (None, None, None, 1 384         conv2d_31[0][0]                  
    __________________________________________________________________________________________________
    batch_normalization_36 (BatchNo (None, None, None, 1 384         conv2d_36[0][0]                  
    __________________________________________________________________________________________________
    activation_31 (Activation)      (None, None, None, 1 0           batch_normalization_31[0][0]     
    __________________________________________________________________________________________________
    activation_36 (Activation)      (None, None, None, 1 0           batch_normalization_36[0][0]     
    __________________________________________________________________________________________________
    conv2d_32 (Conv2D)              (None, None, None, 1 114688      activation_31[0][0]              
    __________________________________________________________________________________________________
    conv2d_37 (Conv2D)              (None, None, None, 1 114688      activation_36[0][0]              
    __________________________________________________________________________________________________
    batch_normalization_32 (BatchNo (None, None, None, 1 384         conv2d_32[0][0]                  
    __________________________________________________________________________________________________
    batch_normalization_37 (BatchNo (None, None, None, 1 384         conv2d_37[0][0]                  
    __________________________________________________________________________________________________
    activation_32 (Activation)      (None, None, None, 1 0           batch_normalization_32[0][0]     
    __________________________________________________________________________________________________
    activation_37 (Activation)      (None, None, None, 1 0           batch_normalization_37[0][0]     
    __________________________________________________________________________________________________
    average_pooling2d_3 (AveragePoo (None, None, None, 7 0           mixed3[0][0]                     
    __________________________________________________________________________________________________
    conv2d_30 (Conv2D)              (None, None, None, 1 147456      mixed3[0][0]                     
    __________________________________________________________________________________________________
    conv2d_33 (Conv2D)              (None, None, None, 1 172032      activation_32[0][0]              
    __________________________________________________________________________________________________
    conv2d_38 (Conv2D)              (None, None, None, 1 172032      activation_37[0][0]              
    __________________________________________________________________________________________________
    conv2d_39 (Conv2D)              (None, None, None, 1 147456      average_pooling2d_3[0][0]        
    __________________________________________________________________________________________________
    batch_normalization_30 (BatchNo (None, None, None, 1 576         conv2d_30[0][0]                  
    __________________________________________________________________________________________________
    batch_normalization_33 (BatchNo (None, None, None, 1 576         conv2d_33[0][0]                  
    __________________________________________________________________________________________________
    batch_normalization_38 (BatchNo (None, None, None, 1 576         conv2d_38[0][0]                  
    __________________________________________________________________________________________________
    batch_normalization_39 (BatchNo (None, None, None, 1 576         conv2d_39[0][0]                  
    __________________________________________________________________________________________________
    activation_30 (Activation)      (None, None, None, 1 0           batch_normalization_30[0][0]     
    __________________________________________________________________________________________________
    activation_33 (Activation)      (None, None, None, 1 0           batch_normalization_33[0][0]     
    __________________________________________________________________________________________________
    activation_38 (Activation)      (None, None, None, 1 0           batch_normalization_38[0][0]     
    __________________________________________________________________________________________________
    activation_39 (Activation)      (None, None, None, 1 0           batch_normalization_39[0][0]     
    __________________________________________________________________________________________________
    mixed4 (Concatenate)            (None, None, None, 7 0           activation_30[0][0]              
                                                                     activation_33[0][0]              
                                                                     activation_38[0][0]              
                                                                     activation_39[0][0]              
    __________________________________________________________________________________________________
    conv2d_44 (Conv2D)              (None, None, None, 1 122880      mixed4[0][0]                     
    __________________________________________________________________________________________________
    batch_normalization_44 (BatchNo (None, None, None, 1 480         conv2d_44[0][0]                  
    __________________________________________________________________________________________________
    activation_44 (Activation)      (None, None, None, 1 0           batch_normalization_44[0][0]     
    __________________________________________________________________________________________________
    conv2d_45 (Conv2D)              (None, None, None, 1 179200      activation_44[0][0]              
    __________________________________________________________________________________________________
    batch_normalization_45 (BatchNo (None, None, None, 1 480         conv2d_45[0][0]                  
    __________________________________________________________________________________________________
    activation_45 (Activation)      (None, None, None, 1 0           batch_normalization_45[0][0]     
    __________________________________________________________________________________________________
    conv2d_41 (Conv2D)              (None, None, None, 1 122880      mixed4[0][0]                     
    __________________________________________________________________________________________________
    conv2d_46 (Conv2D)              (None, None, None, 1 179200      activation_45[0][0]              
    __________________________________________________________________________________________________
    batch_normalization_41 (BatchNo (None, None, None, 1 480         conv2d_41[0][0]                  
    __________________________________________________________________________________________________
    batch_normalization_46 (BatchNo (None, None, None, 1 480         conv2d_46[0][0]                  
    __________________________________________________________________________________________________
    activation_41 (Activation)      (None, None, None, 1 0           batch_normalization_41[0][0]     
    __________________________________________________________________________________________________
    activation_46 (Activation)      (None, None, None, 1 0           batch_normalization_46[0][0]     
    __________________________________________________________________________________________________
    conv2d_42 (Conv2D)              (None, None, None, 1 179200      activation_41[0][0]              
    __________________________________________________________________________________________________
    conv2d_47 (Conv2D)              (None, None, None, 1 179200      activation_46[0][0]              
    __________________________________________________________________________________________________
    batch_normalization_42 (BatchNo (None, None, None, 1 480         conv2d_42[0][0]                  
    __________________________________________________________________________________________________
    batch_normalization_47 (BatchNo (None, None, None, 1 480         conv2d_47[0][0]                  
    __________________________________________________________________________________________________
    activation_42 (Activation)      (None, None, None, 1 0           batch_normalization_42[0][0]     
    __________________________________________________________________________________________________
    activation_47 (Activation)      (None, None, None, 1 0           batch_normalization_47[0][0]     
    __________________________________________________________________________________________________
    average_pooling2d_4 (AveragePoo (None, None, None, 7 0           mixed4[0][0]                     
    __________________________________________________________________________________________________
    conv2d_40 (Conv2D)              (None, None, None, 1 147456      mixed4[0][0]                     
    __________________________________________________________________________________________________
    conv2d_43 (Conv2D)              (None, None, None, 1 215040      activation_42[0][0]              
    __________________________________________________________________________________________________
    conv2d_48 (Conv2D)              (None, None, None, 1 215040      activation_47[0][0]              
    __________________________________________________________________________________________________
    conv2d_49 (Conv2D)              (None, None, None, 1 147456      average_pooling2d_4[0][0]        
    __________________________________________________________________________________________________
    batch_normalization_40 (BatchNo (None, None, None, 1 576         conv2d_40[0][0]                  
    __________________________________________________________________________________________________
    batch_normalization_43 (BatchNo (None, None, None, 1 576         conv2d_43[0][0]                  
    __________________________________________________________________________________________________
    batch_normalization_48 (BatchNo (None, None, None, 1 576         conv2d_48[0][0]                  
    __________________________________________________________________________________________________
    batch_normalization_49 (BatchNo (None, None, None, 1 576         conv2d_49[0][0]                  
    __________________________________________________________________________________________________
    activation_40 (Activation)      (None, None, None, 1 0           batch_normalization_40[0][0]     
    __________________________________________________________________________________________________
    activation_43 (Activation)      (None, None, None, 1 0           batch_normalization_43[0][0]     
    __________________________________________________________________________________________________
    activation_48 (Activation)      (None, None, None, 1 0           batch_normalization_48[0][0]     
    __________________________________________________________________________________________________
    activation_49 (Activation)      (None, None, None, 1 0           batch_normalization_49[0][0]     
    __________________________________________________________________________________________________
    mixed5 (Concatenate)            (None, None, None, 7 0           activation_40[0][0]              
                                                                     activation_43[0][0]              
                                                                     activation_48[0][0]              
                                                                     activation_49[0][0]              
    __________________________________________________________________________________________________
    conv2d_54 (Conv2D)              (None, None, None, 1 122880      mixed5[0][0]                     
    __________________________________________________________________________________________________
    batch_normalization_54 (BatchNo (None, None, None, 1 480         conv2d_54[0][0]                  
    __________________________________________________________________________________________________
    activation_54 (Activation)      (None, None, None, 1 0           batch_normalization_54[0][0]     
    __________________________________________________________________________________________________
    conv2d_55 (Conv2D)              (None, None, None, 1 179200      activation_54[0][0]              
    __________________________________________________________________________________________________
    batch_normalization_55 (BatchNo (None, None, None, 1 480         conv2d_55[0][0]                  
    __________________________________________________________________________________________________
    activation_55 (Activation)      (None, None, None, 1 0           batch_normalization_55[0][0]     
    __________________________________________________________________________________________________
    conv2d_51 (Conv2D)              (None, None, None, 1 122880      mixed5[0][0]                     
    __________________________________________________________________________________________________
    conv2d_56 (Conv2D)              (None, None, None, 1 179200      activation_55[0][0]              
    __________________________________________________________________________________________________
    batch_normalization_51 (BatchNo (None, None, None, 1 480         conv2d_51[0][0]                  
    __________________________________________________________________________________________________
    batch_normalization_56 (BatchNo (None, None, None, 1 480         conv2d_56[0][0]                  
    __________________________________________________________________________________________________
    activation_51 (Activation)      (None, None, None, 1 0           batch_normalization_51[0][0]     
    __________________________________________________________________________________________________
    activation_56 (Activation)      (None, None, None, 1 0           batch_normalization_56[0][0]     
    __________________________________________________________________________________________________
    conv2d_52 (Conv2D)              (None, None, None, 1 179200      activation_51[0][0]              
    __________________________________________________________________________________________________
    conv2d_57 (Conv2D)              (None, None, None, 1 179200      activation_56[0][0]              
    __________________________________________________________________________________________________
    batch_normalization_52 (BatchNo (None, None, None, 1 480         conv2d_52[0][0]                  
    __________________________________________________________________________________________________
    batch_normalization_57 (BatchNo (None, None, None, 1 480         conv2d_57[0][0]                  
    __________________________________________________________________________________________________
    activation_52 (Activation)      (None, None, None, 1 0           batch_normalization_52[0][0]     
    __________________________________________________________________________________________________
    activation_57 (Activation)      (None, None, None, 1 0           batch_normalization_57[0][0]     
    __________________________________________________________________________________________________
    average_pooling2d_5 (AveragePoo (None, None, None, 7 0           mixed5[0][0]                     
    __________________________________________________________________________________________________
    conv2d_50 (Conv2D)              (None, None, None, 1 147456      mixed5[0][0]                     
    __________________________________________________________________________________________________
    conv2d_53 (Conv2D)              (None, None, None, 1 215040      activation_52[0][0]              
    __________________________________________________________________________________________________
    conv2d_58 (Conv2D)              (None, None, None, 1 215040      activation_57[0][0]              
    __________________________________________________________________________________________________
    conv2d_59 (Conv2D)              (None, None, None, 1 147456      average_pooling2d_5[0][0]        
    __________________________________________________________________________________________________
    batch_normalization_50 (BatchNo (None, None, None, 1 576         conv2d_50[0][0]                  
    __________________________________________________________________________________________________
    batch_normalization_53 (BatchNo (None, None, None, 1 576         conv2d_53[0][0]                  
    __________________________________________________________________________________________________
    batch_normalization_58 (BatchNo (None, None, None, 1 576         conv2d_58[0][0]                  
    __________________________________________________________________________________________________
    batch_normalization_59 (BatchNo (None, None, None, 1 576         conv2d_59[0][0]                  
    __________________________________________________________________________________________________
    activation_50 (Activation)      (None, None, None, 1 0           batch_normalization_50[0][0]     
    __________________________________________________________________________________________________
    activation_53 (Activation)      (None, None, None, 1 0           batch_normalization_53[0][0]     
    __________________________________________________________________________________________________
    activation_58 (Activation)      (None, None, None, 1 0           batch_normalization_58[0][0]     
    __________________________________________________________________________________________________
    activation_59 (Activation)      (None, None, None, 1 0           batch_normalization_59[0][0]     
    __________________________________________________________________________________________________
    mixed6 (Concatenate)            (None, None, None, 7 0           activation_50[0][0]              
                                                                     activation_53[0][0]              
                                                                     activation_58[0][0]              
                                                                     activation_59[0][0]              
    __________________________________________________________________________________________________
    conv2d_64 (Conv2D)              (None, None, None, 1 147456      mixed6[0][0]                     
    __________________________________________________________________________________________________
    batch_normalization_64 (BatchNo (None, None, None, 1 576         conv2d_64[0][0]                  
    __________________________________________________________________________________________________
    activation_64 (Activation)      (None, None, None, 1 0           batch_normalization_64[0][0]     
    __________________________________________________________________________________________________
    conv2d_65 (Conv2D)              (None, None, None, 1 258048      activation_64[0][0]              
    __________________________________________________________________________________________________
    batch_normalization_65 (BatchNo (None, None, None, 1 576         conv2d_65[0][0]                  
    __________________________________________________________________________________________________
    activation_65 (Activation)      (None, None, None, 1 0           batch_normalization_65[0][0]     
    __________________________________________________________________________________________________
    conv2d_61 (Conv2D)              (None, None, None, 1 147456      mixed6[0][0]                     
    __________________________________________________________________________________________________
    conv2d_66 (Conv2D)              (None, None, None, 1 258048      activation_65[0][0]              
    __________________________________________________________________________________________________
    batch_normalization_61 (BatchNo (None, None, None, 1 576         conv2d_61[0][0]                  
    __________________________________________________________________________________________________
    batch_normalization_66 (BatchNo (None, None, None, 1 576         conv2d_66[0][0]                  
    __________________________________________________________________________________________________
    activation_61 (Activation)      (None, None, None, 1 0           batch_normalization_61[0][0]     
    __________________________________________________________________________________________________
    activation_66 (Activation)      (None, None, None, 1 0           batch_normalization_66[0][0]     
    __________________________________________________________________________________________________
    conv2d_62 (Conv2D)              (None, None, None, 1 258048      activation_61[0][0]              
    __________________________________________________________________________________________________
    conv2d_67 (Conv2D)              (None, None, None, 1 258048      activation_66[0][0]              
    __________________________________________________________________________________________________
    batch_normalization_62 (BatchNo (None, None, None, 1 576         conv2d_62[0][0]                  
    __________________________________________________________________________________________________
    batch_normalization_67 (BatchNo (None, None, None, 1 576         conv2d_67[0][0]                  
    __________________________________________________________________________________________________
    activation_62 (Activation)      (None, None, None, 1 0           batch_normalization_62[0][0]     
    __________________________________________________________________________________________________
    activation_67 (Activation)      (None, None, None, 1 0           batch_normalization_67[0][0]     
    __________________________________________________________________________________________________
    average_pooling2d_6 (AveragePoo (None, None, None, 7 0           mixed6[0][0]                     
    __________________________________________________________________________________________________
    conv2d_60 (Conv2D)              (None, None, None, 1 147456      mixed6[0][0]                     
    __________________________________________________________________________________________________
    conv2d_63 (Conv2D)              (None, None, None, 1 258048      activation_62[0][0]              
    __________________________________________________________________________________________________
    conv2d_68 (Conv2D)              (None, None, None, 1 258048      activation_67[0][0]              
    __________________________________________________________________________________________________
    conv2d_69 (Conv2D)              (None, None, None, 1 147456      average_pooling2d_6[0][0]        
    __________________________________________________________________________________________________
    batch_normalization_60 (BatchNo (None, None, None, 1 576         conv2d_60[0][0]                  
    __________________________________________________________________________________________________
    batch_normalization_63 (BatchNo (None, None, None, 1 576         conv2d_63[0][0]                  
    __________________________________________________________________________________________________
    batch_normalization_68 (BatchNo (None, None, None, 1 576         conv2d_68[0][0]                  
    __________________________________________________________________________________________________
    batch_normalization_69 (BatchNo (None, None, None, 1 576         conv2d_69[0][0]                  
    __________________________________________________________________________________________________
    activation_60 (Activation)      (None, None, None, 1 0           batch_normalization_60[0][0]     
    __________________________________________________________________________________________________
    activation_63 (Activation)      (None, None, None, 1 0           batch_normalization_63[0][0]     
    __________________________________________________________________________________________________
    activation_68 (Activation)      (None, None, None, 1 0           batch_normalization_68[0][0]     
    __________________________________________________________________________________________________
    activation_69 (Activation)      (None, None, None, 1 0           batch_normalization_69[0][0]     
    __________________________________________________________________________________________________
    mixed7 (Concatenate)            (None, None, None, 7 0           activation_60[0][0]              
                                                                     activation_63[0][0]              
                                                                     activation_68[0][0]              
                                                                     activation_69[0][0]              
    __________________________________________________________________________________________________
    conv2d_72 (Conv2D)              (None, None, None, 1 147456      mixed7[0][0]                     
    __________________________________________________________________________________________________
    batch_normalization_72 (BatchNo (None, None, None, 1 576         conv2d_72[0][0]                  
    __________________________________________________________________________________________________
    activation_72 (Activation)      (None, None, None, 1 0           batch_normalization_72[0][0]     
    __________________________________________________________________________________________________
    conv2d_73 (Conv2D)              (None, None, None, 1 258048      activation_72[0][0]              
    __________________________________________________________________________________________________
    batch_normalization_73 (BatchNo (None, None, None, 1 576         conv2d_73[0][0]                  
    __________________________________________________________________________________________________
    activation_73 (Activation)      (None, None, None, 1 0           batch_normalization_73[0][0]     
    __________________________________________________________________________________________________
    conv2d_70 (Conv2D)              (None, None, None, 1 147456      mixed7[0][0]                     
    __________________________________________________________________________________________________
    conv2d_74 (Conv2D)              (None, None, None, 1 258048      activation_73[0][0]              
    __________________________________________________________________________________________________
    batch_normalization_70 (BatchNo (None, None, None, 1 576         conv2d_70[0][0]                  
    __________________________________________________________________________________________________
    batch_normalization_74 (BatchNo (None, None, None, 1 576         conv2d_74[0][0]                  
    __________________________________________________________________________________________________
    activation_70 (Activation)      (None, None, None, 1 0           batch_normalization_70[0][0]     
    __________________________________________________________________________________________________
    activation_74 (Activation)      (None, None, None, 1 0           batch_normalization_74[0][0]     
    __________________________________________________________________________________________________
    conv2d_71 (Conv2D)              (None, None, None, 3 552960      activation_70[0][0]              
    __________________________________________________________________________________________________
    conv2d_75 (Conv2D)              (None, None, None, 1 331776      activation_74[0][0]              
    __________________________________________________________________________________________________
    batch_normalization_71 (BatchNo (None, None, None, 3 960         conv2d_71[0][0]                  
    __________________________________________________________________________________________________
    batch_normalization_75 (BatchNo (None, None, None, 1 576         conv2d_75[0][0]                  
    __________________________________________________________________________________________________
    activation_71 (Activation)      (None, None, None, 3 0           batch_normalization_71[0][0]     
    __________________________________________________________________________________________________
    activation_75 (Activation)      (None, None, None, 1 0           batch_normalization_75[0][0]     
    __________________________________________________________________________________________________
    max_pooling2d_3 (MaxPooling2D)  (None, None, None, 7 0           mixed7[0][0]                     
    __________________________________________________________________________________________________
    mixed8 (Concatenate)            (None, None, None, 1 0           activation_71[0][0]              
                                                                     activation_75[0][0]              
                                                                     max_pooling2d_3[0][0]            
    __________________________________________________________________________________________________
    conv2d_80 (Conv2D)              (None, None, None, 4 573440      mixed8[0][0]                     
    __________________________________________________________________________________________________
    batch_normalization_80 (BatchNo (None, None, None, 4 1344        conv2d_80[0][0]                  
    __________________________________________________________________________________________________
    activation_80 (Activation)      (None, None, None, 4 0           batch_normalization_80[0][0]     
    __________________________________________________________________________________________________
    conv2d_77 (Conv2D)              (None, None, None, 3 491520      mixed8[0][0]                     
    __________________________________________________________________________________________________
    conv2d_81 (Conv2D)              (None, None, None, 3 1548288     activation_80[0][0]              
    __________________________________________________________________________________________________
    batch_normalization_77 (BatchNo (None, None, None, 3 1152        conv2d_77[0][0]                  
    __________________________________________________________________________________________________
    batch_normalization_81 (BatchNo (None, None, None, 3 1152        conv2d_81[0][0]                  
    __________________________________________________________________________________________________
    activation_77 (Activation)      (None, None, None, 3 0           batch_normalization_77[0][0]     
    __________________________________________________________________________________________________
    activation_81 (Activation)      (None, None, None, 3 0           batch_normalization_81[0][0]     
    __________________________________________________________________________________________________
    conv2d_78 (Conv2D)              (None, None, None, 3 442368      activation_77[0][0]              
    __________________________________________________________________________________________________
    conv2d_79 (Conv2D)              (None, None, None, 3 442368      activation_77[0][0]              
    __________________________________________________________________________________________________
    conv2d_82 (Conv2D)              (None, None, None, 3 442368      activation_81[0][0]              
    __________________________________________________________________________________________________
    conv2d_83 (Conv2D)              (None, None, None, 3 442368      activation_81[0][0]              
    __________________________________________________________________________________________________
    average_pooling2d_7 (AveragePoo (None, None, None, 1 0           mixed8[0][0]                     
    __________________________________________________________________________________________________
    conv2d_76 (Conv2D)              (None, None, None, 3 409600      mixed8[0][0]                     
    __________________________________________________________________________________________________
    batch_normalization_78 (BatchNo (None, None, None, 3 1152        conv2d_78[0][0]                  
    __________________________________________________________________________________________________
    batch_normalization_79 (BatchNo (None, None, None, 3 1152        conv2d_79[0][0]                  
    __________________________________________________________________________________________________
    batch_normalization_82 (BatchNo (None, None, None, 3 1152        conv2d_82[0][0]                  
    __________________________________________________________________________________________________
    batch_normalization_83 (BatchNo (None, None, None, 3 1152        conv2d_83[0][0]                  
    __________________________________________________________________________________________________
    conv2d_84 (Conv2D)              (None, None, None, 1 245760      average_pooling2d_7[0][0]        
    __________________________________________________________________________________________________
    batch_normalization_76 (BatchNo (None, None, None, 3 960         conv2d_76[0][0]                  
    __________________________________________________________________________________________________
    activation_78 (Activation)      (None, None, None, 3 0           batch_normalization_78[0][0]     
    __________________________________________________________________________________________________
    activation_79 (Activation)      (None, None, None, 3 0           batch_normalization_79[0][0]     
    __________________________________________________________________________________________________
    activation_82 (Activation)      (None, None, None, 3 0           batch_normalization_82[0][0]     
    __________________________________________________________________________________________________
    activation_83 (Activation)      (None, None, None, 3 0           batch_normalization_83[0][0]     
    __________________________________________________________________________________________________
    batch_normalization_84 (BatchNo (None, None, None, 1 576         conv2d_84[0][0]                  
    __________________________________________________________________________________________________
    activation_76 (Activation)      (None, None, None, 3 0           batch_normalization_76[0][0]     
    __________________________________________________________________________________________________
    mixed9_0 (Concatenate)          (None, None, None, 7 0           activation_78[0][0]              
                                                                     activation_79[0][0]              
    __________________________________________________________________________________________________
    concatenate (Concatenate)       (None, None, None, 7 0           activation_82[0][0]              
                                                                     activation_83[0][0]              
    __________________________________________________________________________________________________
    activation_84 (Activation)      (None, None, None, 1 0           batch_normalization_84[0][0]     
    __________________________________________________________________________________________________
    mixed9 (Concatenate)            (None, None, None, 2 0           activation_76[0][0]              
                                                                     mixed9_0[0][0]                   
                                                                     concatenate[0][0]                
                                                                     activation_84[0][0]              
    __________________________________________________________________________________________________
    conv2d_89 (Conv2D)              (None, None, None, 4 917504      mixed9[0][0]                     
    __________________________________________________________________________________________________
    batch_normalization_89 (BatchNo (None, None, None, 4 1344        conv2d_89[0][0]                  
    __________________________________________________________________________________________________
    activation_89 (Activation)      (None, None, None, 4 0           batch_normalization_89[0][0]     
    __________________________________________________________________________________________________
    conv2d_86 (Conv2D)              (None, None, None, 3 786432      mixed9[0][0]                     
    __________________________________________________________________________________________________
    conv2d_90 (Conv2D)              (None, None, None, 3 1548288     activation_89[0][0]              
    __________________________________________________________________________________________________
    batch_normalization_86 (BatchNo (None, None, None, 3 1152        conv2d_86[0][0]                  
    __________________________________________________________________________________________________
    batch_normalization_90 (BatchNo (None, None, None, 3 1152        conv2d_90[0][0]                  
    __________________________________________________________________________________________________
    activation_86 (Activation)      (None, None, None, 3 0           batch_normalization_86[0][0]     
    __________________________________________________________________________________________________
    activation_90 (Activation)      (None, None, None, 3 0           batch_normalization_90[0][0]     
    __________________________________________________________________________________________________
    conv2d_87 (Conv2D)              (None, None, None, 3 442368      activation_86[0][0]              
    __________________________________________________________________________________________________
    conv2d_88 (Conv2D)              (None, None, None, 3 442368      activation_86[0][0]              
    __________________________________________________________________________________________________
    conv2d_91 (Conv2D)              (None, None, None, 3 442368      activation_90[0][0]              
    __________________________________________________________________________________________________
    conv2d_92 (Conv2D)              (None, None, None, 3 442368      activation_90[0][0]              
    __________________________________________________________________________________________________
    average_pooling2d_8 (AveragePoo (None, None, None, 2 0           mixed9[0][0]                     
    __________________________________________________________________________________________________
    conv2d_85 (Conv2D)              (None, None, None, 3 655360      mixed9[0][0]                     
    __________________________________________________________________________________________________
    batch_normalization_87 (BatchNo (None, None, None, 3 1152        conv2d_87[0][0]                  
    __________________________________________________________________________________________________
    batch_normalization_88 (BatchNo (None, None, None, 3 1152        conv2d_88[0][0]                  
    __________________________________________________________________________________________________
    batch_normalization_91 (BatchNo (None, None, None, 3 1152        conv2d_91[0][0]                  
    __________________________________________________________________________________________________
    batch_normalization_92 (BatchNo (None, None, None, 3 1152        conv2d_92[0][0]                  
    __________________________________________________________________________________________________
    conv2d_93 (Conv2D)              (None, None, None, 1 393216      average_pooling2d_8[0][0]        
    __________________________________________________________________________________________________
    batch_normalization_85 (BatchNo (None, None, None, 3 960         conv2d_85[0][0]                  
    __________________________________________________________________________________________________
    activation_87 (Activation)      (None, None, None, 3 0           batch_normalization_87[0][0]     
    __________________________________________________________________________________________________
    activation_88 (Activation)      (None, None, None, 3 0           batch_normalization_88[0][0]     
    __________________________________________________________________________________________________
    activation_91 (Activation)      (None, None, None, 3 0           batch_normalization_91[0][0]     
    __________________________________________________________________________________________________
    activation_92 (Activation)      (None, None, None, 3 0           batch_normalization_92[0][0]     
    __________________________________________________________________________________________________
    batch_normalization_93 (BatchNo (None, None, None, 1 576         conv2d_93[0][0]                  
    __________________________________________________________________________________________________
    activation_85 (Activation)      (None, None, None, 3 0           batch_normalization_85[0][0]     
    __________________________________________________________________________________________________
    mixed9_1 (Concatenate)          (None, None, None, 7 0           activation_87[0][0]              
                                                                     activation_88[0][0]              
    __________________________________________________________________________________________________
    concatenate_1 (Concatenate)     (None, None, None, 7 0           activation_91[0][0]              
                                                                     activation_92[0][0]              
    __________________________________________________________________________________________________
    activation_93 (Activation)      (None, None, None, 1 0           batch_normalization_93[0][0]     
    __________________________________________________________________________________________________
    mixed10 (Concatenate)           (None, None, None, 2 0           activation_85[0][0]              
                                                                     mixed9_1[0][0]                   
                                                                     concatenate_1[0][0]              
                                                                     activation_93[0][0]              
    ==================================================================================================
    Total params: 21,802,784
    Trainable params: 21,768,352
    Non-trainable params: 34,432
    __________________________________________________________________________________________________
    

Now since we are using inception model so lets create a inception model using keras and print its layers


{% highlight python linenos %}
layers_contributions=['mixed3', 'mixed5']
{% endhighlight %}

Lets describe layers whose patterns we want to embed into our input image. Here we are using *mixed3* and *mixed5* layers which are concatenation of different convolution layers.


{% highlight python linenos %}
dream_model=deep_dream_model(inception,layers_contributions)
{% endhighlight %}

Now we will create dream model using *deep_dream_model* function which we had defined earlier


{% highlight python linenos %}
deep_outputs=dream_model(preprocessed_image)
for layer_name,outputs in zip(layers_contributions,deep_outputs):
    print(layer_name)
    print(outputs.shape)
    print(outputs.numpy().mean())
{% endhighlight %}

    mixed3
    (1, 19, 30, 768)
    0.44434533
    mixed5
    (1, 19, 30, 768)
    0.16857202
    

Lets test how we can extract and manipulate activations of layers which we have defined in *layers_contributions* using our deep dream model


{% highlight python linenos %}
model_output= lambda model,inputs:model(inputs)
{% endhighlight %}

Now lets define a helper function which will return output of model on providing input. Above we have defined a *lambda* function which takes model and input image as parameter and return output of model *ie..*  activations of layers which we have defined in *layers_contributions*


{% highlight python linenos %}
def get_loss(activations):
    loss=[]
    for activation in activations:
        loss.append(tf.math.reduce_mean(activation))
    return tf.reduce_sum(loss)
{% endhighlight %}

the above function defines our loss function which we will maximize using gradient ascent. It is simply sum of mean of activations


{% highlight python linenos %}
def get_loss_and_gradient(model,inputs,total_variation_weight=0):
    with tf.GradientTape() as tape:
        tape.watch(inputs)
        activations=model_output(model,inputs)
        loss=get_loss(activations)
        loss=loss+total_variation_weight*tf.image.total_variation(inputs)
    grads=tape.gradient(loss,inputs)
    grads /= tf.math.reduce_std(grads) + 1e-8 
    return loss,grads
{% endhighlight %}

the above function returns gradient (derivative) of our loss function with respect to our input image. We use tensorflow *GradientTape* to calculate gradients. First we have to watch our input image since it is not tensorflow variable then we get our model outputs which are activations of layers in *layers_contributions* and we will use these activations to find loss and finally find out gradient using *tape.gradient* method we also standardized our gradients by dividing it with standard deviation of gradients. A small number *1e-8* is also added to prevent accidentally division by 0.

There is also *total_variation_weight* parameter this will be used for adding some amount of *total_variation* loss into our loss function.

The total variation loss is the sum of the absolute differences for neighbouring pixel-values in the input images. This measures how much noise is in the images.

*total variation loss* is not necessary for deep dream outputs but can be used to smooth out result. play with *total_variation_weight* parameter to find result of your liking.


{% highlight python linenos %}
def run_gradient_ascent(model,inputs,epochs=1,steps_per_epoch=1,weight=0.01,total_variation_weight=0):
    img = tf.convert_to_tensor(inputs)
    start=time.time()
    for i in range(epochs):
        print(f"epoch: {i+1}",end=' ')
        for j in range(steps_per_epoch):
            loss,grads=get_loss_and_gradient(model,img,total_variation_weight)
            img = img + grads*weight
            img = tf.clip_by_value(img, -1.0, 1.0)
            print('=',end='')
        print("\n")
    end=time.time()
    print(f"Time elapsed: {end-start:1f}sec")
    return img.numpy()
{% endhighlight %}

Now we have gradients of loss with respect to input image, we define a function that will do gradient ascent by changing our input image in direction of gradients. This will maximize input space which will eventually increase activations of layers.
This function also takes *epochs* as parameter which is number of iteration for which we want to process image. *weight* parameter is strength of patterns embeds to image. Method also prints stats of each epoch


{% highlight python linenos %}
image_array=run_gradient_ascent(dream_model,preprocessed_image,epochs=2,steps_per_epoch=100,weight=0.01)
{% endhighlight %}

    epoch: 1 ====================================================================================================
    
    epoch: 2 ====================================================================================================
    
    Time elapsed: 62.779686sec
    

Now its time to create deep dream image, we apply gradient ascent for some epochs and save our image into a variable which is a numpy array


{% highlight python linenos %}
show_image(deprocess_inception_image(image_array))
resultant_image=array_to_img(image_array,True)
resultant_image.save("deep_dream_simple.jpg")
{% endhighlight %}


![png](https://storage.googleapis.com/tarun-bisht.appspot.com/blogs/deep_dream_3e52bb8d7841697e1)


Now we are ready to see how our input image looks like. We first deprocess our numpy array and then convert it to image and finally save output image to hard drive as image.

## Deep Dreaming using octaves

To improve quality of patterns in image we can use octaves technique. In this technique input image is processed at different scale. Each different size image is an octave this improve quality of patterns on image.

### Steps:
- first base shape of image is saved to a variable
- input image is then scaled to different sizes smaller and greater than base shape
- these octaves (different scaled images) are then passed to *run_gradient_ascent* function to apply gradient ascent to each octave.
- finally resultant image is again resized to base shape


{% highlight python linenos %}
def run_gradient_ascent_with_octaves(model,inputs,epochs=1,steps_per_epoch=1,num_octaves=2,octave_size=1.3,weight=0.01,total_variation_weight=0):
    img=tf.convert_to_tensor(inputs)
    assert len(inputs.shape)<=4 or len(inputs.shape)>=3
    if len(inputs.shape)==3:
        base_shape=img.shape[:-1]
    base_shape=img.shape[1:-1]
    for n in range(-num_octaves,1):
        print(f'Processing Octave: {n*-1}')
        new_shape=tuple([int(dim * (octave_size**n)) for dim in base_shape])
        img=tf.image.resize(img,new_shape)
        img=run_gradient_ascent(model,img,epochs,steps_per_epoch,weight,total_variation_weight)
    return tf.image.resize(img,base_shape).numpy()
{% endhighlight %}

the above function runs gradient ascent using octave technique. It takes *num_octaves* parameter which is number of octaves you want to process. Default is 2 that means it process 2 octaves and 1 original image. 

Image is resized using tensorflow *image.resize* function. New shape is calculated by raising height and width of image to power of octave number to process. As you can notice loops starts from *-num_octaves to 0 (excluding 1)*. Negative power will scale down the image from its base shape. We can also run loop from *-num_octaves to +num_octave* but as image size increases it consume more RAM. 

*octave_size* parameter tells by what factor we want to scale images 


{% highlight python linenos %}
image_array=run_gradient_ascent_with_octaves(dream_model,preprocessed_image,epochs=1,steps_per_epoch=100,num_octaves=3,octave_size=1.3,weight=0.01)
{% endhighlight %}

    Processing Octave: 3
    epoch: 1 ====================================================================================================
    
    Time elapsed: 25.796944sec
    Processing Octave: 2
    epoch: 1 ====================================================================================================
    
    Time elapsed: 27.048554sec
    Processing Octave: 1
    epoch: 1 ====================================================================================================
    
    Time elapsed: 27.536671sec
    Processing Octave: 0
    epoch: 1 ====================================================================================================
    
    Time elapsed: 31.441849sec
    

Now its time to create deep dream image, It takes more time to create deep dream but it worth.


{% highlight python linenos %}
show_image(deprocess_inception_image(image_array))
image=array_to_img(image_array,True)
image.save("deep_dream_with_octave.jpg")
{% endhighlight %}


![png](https://storage.googleapis.com/tarun-bisht.appspot.com/blogs/deep_dream_4c563e15f644beece)


And this time we got some exciting results.

## Deep Dreaming using Image Tiling

As we start processing bigger images we need more RAM to put it into memory for calculating gradients. Also we cannot process more octaves using above techniques. 

This issue can be fixed by using image tilings, In this technique we split image into tiles and gradient is calculated for each tile seperately. 

By tiling images into small sizes and processing these tiles solves the issue as we have to process small tiles of image not the entire image.

While tiling we make sure that it is random else we get seam in our image after processing.


{% highlight python linenos %}
# Randomly rolls the image to avoid tiled boundaries.
def random_image_tiling(img, maxdim):
    shift = tf.random.uniform(shape=[2], minval=-maxdim, maxval=maxdim, dtype=tf.int32)
    shift_r,shift_d=shift
    img_rolled = tf.roll(img, shift=[shift_r,shift_d], axis=[1,0])
    return shift_r, shift_d, img_rolled
{% endhighlight %}

the above function takes image as input and randomly rolls the image to avoid tiled boundaries. It returns shifted image and positions from where image was shifted. We have used tensorflow *roll* function to shift images. It create roll of an array along different axis from shift positions specified.


{% highlight python linenos %}
shift_r,shift_d,img_tiled=random_image_tiling(input_image[0], 512)
show_image(img_tiled.numpy())
{% endhighlight %}


![png](https://storage.googleapis.com/tarun-bisht.appspot.com/blogs/deep_dream_5ccaac83447e81783)


lets test of how random tiled image function transforms our image and randomly roll it so that we do not get same tile everytime we process and seam across image.


{% highlight python linenos %}
def get_loss_and_grads_with_tiling(model,inputs,tile_size=512,total_variation_weight=0.004):
    shift_r,shift_d,rolled_image=random_image_tiling(inputs[0],tile_size)
    grads=tf.zeros_like(rolled_image)
    # create a tensor from 0 to rolled_image width with step of tile size
    x_range = tf.range(0, rolled_image.shape[0], tile_size)[:-1]
    # check if x_range is not empty
    if not tf.cast(len(x_range), bool):
        x_range= tf.constant([0])
    # create a tensor from 0 to rolled_image height with step of tile size
    y_range = tf.range(0, rolled_image.shape[1], tile_size)[:-1] 
    # check if y_range is not empty
    if not tf.cast(len(y_range), bool):
        y_range=tf.constant([0])
    for x in x_range:
        for y in y_range:
            with tf.GradientTape() as tape:
                tape.watch(rolled_image)
                # here we create tile from rolled image of size=tile_size
                image_tile= tf.expand_dims(rolled_image[x:x+tile_size, y:y+tile_size],axis=0)
                activations=model_output(model,image_tile)
                loss=get_loss(activations)
                loss=loss+total_variation_weight*tf.image.total_variation(image_tile) 
            grads=grads+tape.gradient(loss,rolled_image)
    grads = tf.roll(grads, shift=[-shift_r,-shift_d], axis=[1,0]) #reverse shifting of rolled image
    grads /= tf.math.reduce_std(grads) + 1e-8
    return loss,grads
{% endhighlight %}

Lets define a way to get gradients from tiled image. In above function we first get random rolled image and its rolling positions using *random_image_tiling* function. We then have some logic to create a tile from rolled image of size *tile_size* specified. This tile image is then passed to model and loss is calculated finally gradients are calculated for that tile and added to *grads* tensor. We process small tiles of rolled image till we have iterated whole image (rolled image) and all gradients of each tile are summed together. Then we reverse the shiftings of rolled image back to original image and finally scaling and returning the gradients.


{% highlight python linenos %}
def run_gradient_ascent_with_octave_tiling(model,inputs,steps_per_octave=100,num_octaves=2,octave_size=1.3,tile_size=512,weight=0.01,total_variation_weight=0.0004):
    img=tf.convert_to_tensor(inputs)
    weight=tf.convert_to_tensor(weight)
    assert len(inputs.shape)<=4 or len(inputs.shape)>=3
    if len(inputs.shape)==3:
        base_shape=img.shape[:-1]
    base_shape=img.shape[1:-1]
    start=time.time()
    for n in range(-num_octaves,num_octaves+1):
        print(f'Processing Octave: {n+num_octaves+1}')
        new_shape=tuple([int(dim*(octave_size**n)) for dim in base_shape])
        img=tf.image.resize(img,new_shape)
        for step in range(steps_per_octave):
            print('=',end='')
            loss,grads=get_loss_and_grads_with_tiling(model,img,tile_size,total_variation_weight)
            img = img + grads*weight
            img = tf.clip_by_value(img, -1.0, 1.0)
        print("\n")
    end=time.time()
    print(f"Time elapsed: {end-start:.1f} sec")
    return tf.image.resize(img, base_shape).numpy()
{% endhighlight %}

the above funtion is same as *run_gradient_ascent_with_octave* but instead of using *get_loss_and_grads* function here we have used *get_loss_and_grads_with_tiling* to get gradients using tiling images strategy.


{% highlight python linenos %}
image_array=run_gradient_ascent_with_octave_tiling(dream_model,preprocessed_image,steps_per_octave=100,num_octaves=3,octave_size=1.3,tile_size=512,weight=0.01,total_variation_weight=0)
{% endhighlight %}

    Processing Octave: 1
    ====================================================================================================
    
    Processing Octave: 2
    ====================================================================================================
    
    Processing Octave: 3
    ====================================================================================================
    
    Processing Octave: 4
    ====================================================================================================
    
    Processing Octave: 5
    ====================================================================================================
    
    Processing Octave: 6
    ====================================================================================================
    
    Processing Octave: 7
    ====================================================================================================
    
    Time elapsed: 270.3 sec
    

Now its time to create deep dream image. Time taken to create dream depends on size of input image passed. It also uses octave technique previously discussed to improve quality of image but now we can process more octaves.


{% highlight python linenos %}
show_image(deprocess_inception_image(image_array))
image=array_to_img(image_array,True)
image.save("deep_dream_with_octave_tiling.jpg")
{% endhighlight %}


![png](https://storage.googleapis.com/tarun-bisht.appspot.com/blogs/deep_dream_6f2e0c8cf112caee2)


and this time we got some more exciting results also we can create bigger resolution dream.

So finally we are ready to see some deep dreams of neural networks. Play with it and share exciting results.

[![deepdream result videp](http://img.youtube.com/vi/wmDjFQDh5BY/0.jpg)](http://www.youtube.com/watch?v=wmDjFQDh5BY)

Thanks for reading till last. ✌✌✌

[IPython Notebook Link](https://github.com/tarun-bisht/blogs-notebooks/tree/master/deepdream)

### References

[Tensorflow Tutorials](https://www.tensorflow.org/tutorials/generative/deepdream)
[Keras Book](https://livebook.manning.com/book/deep-learning-with-python/chapter-8/76)