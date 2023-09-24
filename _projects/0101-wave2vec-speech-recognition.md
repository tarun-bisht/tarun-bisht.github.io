---
layout: writing
title: Wav2Vec2 ASR
category: AI/ML
icon: /assets/projects/wav2vec2/icon.png
tags: asr speech-recognition speech-to-text transcribe-audio
comment: true
urls:
  github: https://github.com/tarun-bisht/wav2vec2-asr
---

This project uses wav2vec2 model introduced by facebook AI in their paper [wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations
](https://arxiv.org/abs/2006.11477). The project uses implementations of wav2vec2 from hugging face transformers to create an ASR system which takes input speech signal as input and outputs transcriptions asynchronously. This project also includes training notebooks to train your own speech recognition system.

I have also written a [post](https://www.tarunbisht.com/deep%20learning/2021/06/17/speech-recognition-using-wav2vec-model/) explaining wave2vec2 in some detail with some further learning directions.

## Installation

#### Get Started

- Install Python3 or anaconda and install them. For detailed steps follow installation guide for [Python3](https://realpython.com/installing-python/) and [Anaconda](https://docs.anaconda.com/anaconda/install/)
- Install required packages via pip or conda.

#### Installing via pip

- Download and Install python
- Create a virtual environment using `python -m venv env_name`
- enable created environment `env_path\Scripts\activate`
- Install PyTorch `pip install torch==1.8.0+cu102 torchaudio===0.8.0 -f https://download.pytorch.org/whl/torch_stable.html`
- Install required dependencies `pip install -r requirements.txt`

#### Installing via conda

- Download and install miniconda
- Create a new virutal environment using `conda create --name env_name python==3.8`
- enable create environment `conda activate env_name`
- Install PyTorch `conda install pytorch torchaudio cudatoolkit=11.1 -c pytorch`
- Install required dependencies `pip install -r requirements.txt`

### Usage Instructions

- Download Github repository
- Follow [README guide](https://github.com/tarun-bisht/wav2vec2-asr/blob/master/README.md#inferencing) for using the application.

## Tested Platforms

- native windows 10 ✔
- windows-10 wsl2 cpu ✔
- windows-10 wsl2 gpu ✔
- Linux ✔
