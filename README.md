# AdEMAMix Optimizer Implementation

## Introduction
This repository provides an implementation of the AdEMAMix optimizer, which is designed to improve upon the traditional Adam optimizer by utilizing a mixture of two Exponential Moving Averages (EMAs). AdEMAMix was proposed in the paper [The AdEMAMix Optimizer: Better, Faster, Older](https://arxiv.org/abs/2409.03137), and it leverages both recent and older gradients to enhance model convergence, particularly in complex architectures like Convolutional Neural Networks (CNNs) and Transformer models.

## About AdEMAMix
AdEMAMix is a novel optimizer that modifies the standard Adam optimizer by incorporating two EMA momentum terms:
- Fast EMA (m1): Tracks recent gradients, similar to Adam's momentum term.
- Slow EMA (m2): Tracks older gradients, allowing the optimizer to benefit from long-term gradient accumulation.

This combination allows AdEMAMix to be responsive to local changes in the loss landscape while leveraging historical gradients to improve generalization and stability.

Example implementaion can be found in jupyter notebook test_ademamix.ipynb
