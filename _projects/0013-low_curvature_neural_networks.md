---
layout: writing
title: Low Curvature Neural Nets
category: Course-Projects AI/ML Research-Projects
icon: /assets/projects/LCNN/icon.png
tags: deep-learning machine-learning autoencoders MCMC
comment: true
math: true
urls:
  github: https://github.com/tarun-bisht/lcnn
---

This project is based on the paper [ Efficient Training of Low-Curvature Neural Networks](https://openreview.net/forum?id=2B2xIJ299rx){:target="\_blank"} by _Suraj Srinivas, Kyle Matoba, Himabindu Lakkaraju, & François Fleuret_

This project studies a new approach called low-curvature neural networks (LCNNs) to tackle issues such as low adversarial robustness and gradient instability in standard deep neural networks. LCNNs demonstrate lower curvature compared to standard models while maintaining similar predictive performance, leading to improved robustness and stable gradients. The authors decompose overall model curvature in terms of the constituent layer’s curvatures and slopes. They also introduce two novel architectural components: the centered-softplus non-linearity and a Lipschitz-constrained batch normalization layer. We compare the performance and adversarial robustness of LCNNs in the MNIST dataset. We then propose parametric swish non-linearity. Our experiments showed that centered-softplus proposed by the author is better than our proposed parametric swish.

## Experiments

(1) evaluate performance of LCNNs

(2) compare the adversarial robustness of LCNNs

(3) evaluate the effectiveness of our proposed parameterized swish activation and its comparison with centered softplus.

A simple classifier with 2 convolutional layer + maxpooling and 2 linear layers with dropout was used.

## Other Links

- [Project Report]({% link /assets/projects/LCNN/final_report.pdf %}){:target="\_blank"}
- [Project PPT Part 1]({% link /assets/projects/LCNN/PPT_part1.pdf %}){:target="\_blank"}
- [Project PPT Part 2]({% link /assets/projects/LCNN/PPT_part2.pdf %}){:target="\_blank"}
- [Experiment Python Notebook]({% link /assets/projects/LCNN/LCNN_Experiment.ipynb %}){:target="\_blank"}

## References

- Suraj Srinivas, Kyle Matoba, Himabindu Lakkaraju, and François Fleuret. Efficient training of low-curvature neural networks. Advances in Neural Information Processing Systems, 2022.
