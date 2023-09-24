---
layout: writing
title: Langevin Autoencoders
category: Course Projects
icon: /assets/projects/LAE/icon.png
tags: deep-learning machine-learning autoencoders MCMC
comment: true
math: true
urls:
  github: https://github.com/tarun-bisht/IE506-Project-2023
---

This project is based on the paper [Langevin Autoencoders for Learning Deep Latent Variable Models](https://arxiv.org/abs/2209.07036){:target="\_blank"} by _Shohei Taniguchi, Yusuke Iwasawa, Wataru Kumagai, Yutaka Matsuo_

This project report aims to train a deep latent variable model using Markov Chain Monte Carlo (MCMC) by simulating a stochastic differential equation called Langevin dynamics. This project studies Langevin Autoencoder (LAE), which uses Amortized Langevin dynamics (ALD) that replaces datapoint-wise MCMC iteration with updates on an encoder that maps data points into latent variables. Using the MNIST dataset, we compare the generation capability and latent space of LAE and Variational Autoencoder (VAE). We also benchmark different parameters of LAE and their effect. We then proposed a new training algorithm based on deep latent variable models (DLVMs) that incorporates DLVM while training the model to generate similar data samples. We benchmarked our proposed training method and show that this training procedure is more robust towards adversarial attacks than normal training.

## Experiments

(1) evaluate the generation capability of LAE and compare it to VAE

(2) compare the latent space of VAE and LAE

(3) evaluate the representation learning capability of LAE and compare it 5with VAE

(4) evaluate the effectiveness of our proposed training method

(5) evaluate whether our proposed training method is robust toward adversarial attacks

(6) generation of samples from interpolation of latent space.

These experiments are primarily conducted on linear encoder and decoder using the MNIST datasets and the PyTorch torch distributions framework.

## Other Links

- [Project Report]({% link /assets/projects/LAE/final_report.pdf %}){:target="\_blank"}
- [Project PPT Part 1]({% link /assets/projects/LAE/PPT_part1.pdf %}){:target="\_blank"}
- [Project PPT Part 2]({% link /assets/projects/LAE/PPT_part2.pdf %}){:target="\_blank"}

## References

- Pytorch distributions, 2023.
- Shohei Taniguchi, Yusuke Iwasawa, Wataru Kumagai, and Yutaka Matsuo. Langevin autoencoders for learning deep latent variable models, 2022.
