---
layout: writing
title:  "Speech Recognition using wav2vec2 : Speech to Text"
date:   2021-06-17 22:16:50 +0530
category: Deep Learning
tags: python speech ASR intermediate pytorch wav2vec wave2vec2 facebookAI
description: Speech recognition is a task where, machine listens to speech input, understand it and outputs what it recognise. One such task is speech to text where machine listens to human speech and output transcription of that speech input. Speech recognition has introduced a new way to interact with machines which is a lot more natural for humans. Today, we have seen lots of voice-enabled devices around us, Siri, Google Assistant, Cortana are voice-enabled assistants which we encounter in our daily life. In this post, we will learn and explore wav2vec for speech recognition.
comment: true
math: true
---
![Photo by <a href="https://unsplash.com/@danielsandvik?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Daniel Sandvik</a> on <a href="https://unsplash.com/s/photos/speech?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Unsplash</a>]({% link /assets/blogs/wav2vec/microphone.jpg %})

Speech recognition is a task where, machine listens to speech input, understand it and outputs what it recognise. One such task is speech to text where machine listens to human speech and output transcription of that speech input. Speech recognition has introduced a new way to interact with machines which is a lot more natural for humans. Today, we have seen lots of voice-enabled devices around us, Siri, Google Assistant, Cortana are voice-enabled assistants which we encounter in our daily life. In this post, we will learn and explore wav2vec for speech recognition.

wav2vec is a speech representation learning framework from Facebook AI. It learns speech representations from the raw audio signal using self-supervised learning. Before moving forward in the post let's first understand the concept of speech representation and self-supervised learning. I will also point to other resources for some topics as those topics are vast and cannot be explained in a single post.

Speech representations are features that help to understand the context in speech audio. An infant is not taught how to understand spoken words by his parents, it learns to understand speech on its own by listening to adults around. This process requires learning good representations from speech data. In machine learning these representations are in form of vectors that are output from a feature extractor when an audio signal is passed into it.

Self-supervised learning is a way to learn representations from data by using supervisory signals from data. It has already shown its potential in the field of NLP with the introduction of models like BERT and GPT. Once a model has learned useful representations from data we can use this pretrained model in various predictive tasks or finetune it by adding other layers in top based on our use-case. For example, in NLP if we have the task of sentimental analysis, we can use the BERT model and add additional classification layers on top of it. BERT is used to extract useful representations from the input text and we use these representations to classify positive or negative sentence. Also to do finetuning, we need less amount of labeled data as we are not training from scratch, our model already knows how to represent text data. Other use cases include zero-shot classification where we can perform nearest neighbour or clustering in feature space(features we get from pretrained model). Researchers are exploring use-cases and different ways to implement it in computer vision, scene understanding, robotics etc.

There are number of ways to pass these supervisory signals to model one of the way is to use some sort of objective or pretext task. These tasks are learned by models to build useful representations from data. In BERT we use pretext task similar to fill in the blanks, where some input words are masked and the model is trained to predict these masked words. This way model learns the relationship between different words and sorts out which word is more probable to come. In vision, we have task like predicting rotation of images, where we rotate our input image by any random discrete angle (0, 90, 180, 270) and predict the angle in which the image is rotated. This way model can learn features like the sky is always up, trees are grown upward etc.

In the representation learning framework, we have two steps, first step is to learn representations from data utilizing self-supervised learning from unlabeled data, then in the second step use these representations to finetune with some amount of labelled data. In NLP this framework was very successful, hence authors of wav2vec model have tried and extended this framework in speech recognition task.

This wav2vec framework has multiple versions with improvements in every later version. Currently, we have wav2vec2 as the latest version, which will be the focus of the post. To prepare for this let's briefly explore previous wav2vec versions.

## wav2vec

It is a convolutional neural network that takes raw audio as input and output representation of speech (vector). The model is trained using a contrastive loss where the model aims to predict future frames from a set of positive and negative samples. We want the model future frame prediction similar to the true future frame(positive sample) and dissimilar to false future frames(negative samples). The model used here is an autoencoder, first raw audio sample $$X$$ is mapped to a compressed representation $$Z$$ ie. $$f:X \mapsto Z$$ and then the compressed representation is mapped to the representation vector $$C$$ ie. $$g:Z \mapsto C$$.
>$$X \mapsto Z \mapsto C$$.

Refer to these resources to learn more about contrastive learning:

- [Contrastive Representation Learning by Lilianweng](https://lilianweng.github.io/lil-log/2021/05/31/contrastive-representation-learning.html)
- [Triplet Loss by Andrew Ng](https://www.youtube.com/watch?v=d2XB5-tuCWU)
- [Understanding Contrastive Learning](https://towardsdatascience.com/understanding-contrastive-learning-d5b19fd96607)

## vq-wav2vec

This version added vector quantization in the encoder's output to discretized speech representation. Since human speech is represented in discrete phoneme and language, this idea of quantization was used. The idea is similar to VQ-VAE(vector quantized variational autoencoder). The compressed representation from encoder $$f:X \mapsto Z$$ is quantized using Gumbel-Softmax, which is a differentiable way to sample discrete data and output discrete representation $$q: Z \mapsto Q$$.

We use codebook to convert continuous representations into discrete representations. Codebooks are set of vectors to choose quantized representation from. We can look at these codebooks as different categories to choose from based on softmax values.

The Gumbel-softmax enables chossing discrete codebook entries in differentiable way. The feature encoder output $$z$$ is mapped to $$l \epsilon R^{G \times V}$$, where $$R^{G \times V}$$ is set of all codebook vectors, $$G$$ is number of codebooks, $$V$$ is number of entries in a codebook and $$l$$ is discrete vector choosen from codebooks for representation $$z$$. The probability to choose a vector entry $$v$$ from codebook $$g$$, is defined as:

$$p_{g,v} = \frac{exp(l_{g,v} + n_{v})/\tau}{\sum_{k=1}^{v}exp(l_{g,k}+n_k)}$$

where, $$\tau$$ is non negative temperature hyperparameter, $$n=-log(-log(u))$$, $$u$$ are uniform samples from continious uniform distribution.

These discrete representations are passed to a context network (BERT in this case) that maps them to representation vector $$C$$ ie. $$g:Q \mapsto C$$.
>$$X \mapsto Z \mapsto Q \mapsto C$$.

The training objective in vq-wav2vec is similar to wav2vec.

Refer to these resources for details:

- [Vector Quantization for Machine Learning, explains codebook concept](https://machinelearningmastery.com/learning-vector-quantization-for-machine-learning/)
- [Gumbel Softmax](https://neptune.ai/blog/gumbel-softmax-loss-function-guide-how-to-implement-it-in-pytorch)
- [VQ-VAE explained](https://ml.berkeley.edu/blog/posts/vq-vae/)

## wav2vec2

The self-supervised objective for wav2vec2 is similar to BERT, at first raw audio signal is passed from a set of a convolutional network which encodes speech data and then we mask a certain proportion of time steps from this latent feature encoder space(output from convolutional feature encoder) and train model to identify correct quantized latent audio representation for each masked time step. The final model is then fine-tuned on a labeled dataset.

![wav2vec model]({% link /assets/blogs/wav2vec/wav2vec_model.svg %})

The model is composed of a multi-layer convolutional feature encoder, which takes input raw audio, and outputs a latent representation for T time-steps $$f:X \mapsto Z$$. They are fed into a transformer module to build representations using the entire sequence $$g:Z \mapsto C$$. The output from feature encoder $$f$$ is discretized in parallel using quantization that uses product quantization $$q: Z \mapsto Q$$. These quantized representations, represent targets in self-supervised objective.

The product quantization amounts to choosing quantized representation from multiple codebooks and concatenating them. This output is then passed to Gumbel-softmax to get quantized representation.

![wav2vec model]({% link /assets/blogs/wav2vec/wav2vec_train.svg %})

The training objective here is to make masked representation from context network similar to the quantized representation of that masked timestep (positive sample) and dissimilar to the quantized vector of other parts of time step (negative samples). The loss function used here is the combination of contrastive loss and diversity loss.

### Loss Function

$$L = L_{contrastive} + \alpha \times L_{diversity}$$

Loss function is composed of contrastive loss weighted by diversity loss. The model learns with contrastive loss using quantized representation of positive and negative samples. Diversity loss is applied to encourage the model to use the codebook entries equally. $$\alpha$$ is hyperparameter to weight the use of diversity loss.

$$L_{contrastive} = -log \frac{exp(sim(c_{t}, q_{t})/\tau)}{\sum_{\bar{q}\epsilon Q(t)} exp(sim(c_{t}, \bar{q}))}$$

here, $$c_{t}$$ is output from context network for masked time step t, $$q_{t}$$ is quantized representation for positive sample (masked quantized representation at timestep t) and $$\bar{q}$$ is quantized representation for negative samples. $$sim(a,b)$$ is scaled cosine similarity between $$a$$ and $$b$$.

$$sim(a,b) = \frac{a.b}{|a||b|}$$

The final trained model is then finetuned for speech recognition by adding a randomly initialized linear layer on top of the context vector. The number of units in the added linear layer is 29, which is equal to the number of tokens for character targets plus a word boundary token

## Using wav2vec2 for speech recognition

We will use the huggingface library’s implementation of the wav2vec model in this project. This model works on non-streaming speech data.

Lets start by installing transformers by huggingface

{% highlight bash linenos %}
pip install -q transformers
{% endhighlight %}

Importing required dependencies. librosa is used for loading speech data and processing, torch for tensor manipulation and transformers Wav2Vec2Tokenizer and Wav2Vec2ForCTC. Wav2Vec2Tokenizer tokenizes input speech and Wav2Vec2ForCTC is wav2vec model for speech recognition.

{% highlight python linenos %}
import librosa
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer
{% endhighlight %}

Now lets instantiate pretrained wav2vec model and tokenizer, from hugging face repo.

{% highlight python linenos %}
# load model and tokenizer
tokenizer = Wav2Vec2Tokenizer.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
{% endhighlight %}

Here we load speech data using librosa, make sure load audio with sample rate 16KHz, as pretrained model was trained with the same sample rate.

{% highlight python linenos %}
speech, rate = librosa.load("speech.wav", sr=16000)
{% endhighlight %}

Now, we will tokenize input speech using tokenizer and pass it to wav2vec model.

{% highlight python linenos %}
input_values = tokenizer(speech, return_tensors = 'pt').input_values
# logits (non-normalized predictions)
logits = model(input_values).logits
{% endhighlight %}

Finally, use argmax to get index of highest probability character token and decode using tokenizer decode method.

{% highlight python linenos %}
predicted_ids = torch.argmax(logits, dim =-1)
# decode the audio to generate text
transcriptions = tokenizer.decode(predicted_ids[0])
print(transcriptions)
{% endhighlight %}

I hope this post explains wav2vec model or point towards right direction to learn about it. Thanks for reading till last.

## Further Readings

- [BERT paper](https://arxiv.org/abs/1810.04805)
- [wav2vec paper](https://arxiv.org/abs/1904.05862)
- [wav2vec2 paper](https://arxiv.org/abs/2006.11477)
- [Transformer architecture explained](https://www.youtube.com/watch?v=OyFJWRnt_AY)
- [Attention explained](https://www.youtube.com/watch?v=d25rAmk0NVk)
- [Self-Supervised Learning by Lilianweng](https://lilianweng.github.io/lil-log/2019/11/10/self-supervised-learning.html)
- [Wav2Vec Post by Maël Fabien](https://maelfabien.github.io/machinelearning/wav2vec/)
