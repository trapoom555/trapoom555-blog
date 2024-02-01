---
author: "trapoom555"
title: "RT-1 : Robot Transformer 1"
date: "2024-01-31"
math: mathjax
tags: [
    "multimodal",
    "robot-learning",
    "manipulator",
    "paper-review",
    "computer-vision",
    "NLP",
    "transfer-learning"
]
---

## Framework

RT-1 [[1]](#1) takes images and language instructions and outputs discretized base and arm actions to control mobile manipulator in a close-loop manner. The framework not only tries to outperform other baselines performance but also aims to reach **real-time** execution as much as possible.

## Architecture

RT-1 combines efficient models together to build an efficient model with 35M parameters. The models include FiLM [[2]](#2), EfficientNet-B3 [[3]](#3), Universal Sentence Encoder [[4]](#4), TokenLearner [[5]](#5) and [Transformer](https://trapoom555.github.io/trapoom555-blog/posts/transformer/) [[6]](#6). This combination can be performed in real environment at $3\ \text{Hz}$.

| <img src="https://github.com/trapoom555/trapoom555-blog/blob/main/static/images/RT1/architecture.png?raw=true" style= "display: block; margin-left: auto; margin-right: auto; width: 80%;"/>|
|:--:| 
| *RT-1 Architecture (Image from [[1]](#1))* |

### Image Encoder

EfficientNet-B3 takes history of 6 images with size of $300 \times 300$ pixels. Each image will give an output feature map with size of $9\times 9 \times 512$. The feature map of each image will be then flattened as $81$ visual tokens.

### Language Encoder

Universal Sentence Encoder (USE) encodes natural language instruction to a fixed-size token with the dimension of $512$.

### Condition Image Tokens with Language

Fusing language tokens information to a imageNet pre-trained EfficientNet-B3 image model can be achieved by inserting FiLM layers to *inner layers* of the image model. In this case, they are added between MBConv layers of the EfficientNet-B3. The FiLM layers are initialized in the way that at the beginning of training it acts like an identity which preserve the output of pre-trained image model.

### TokenLearner

TokenLearner acts as a soft-selector which compress 81 image tokens to 8 output tokens based on their information. This compression makes the model more efficient at infrerence time.

### Transformer

Feeding 6 x 8 image tokens (there're 6 images in the history) to a Transformer will produce action tokens as outputs. The Transformer is a decoder-only sequence model with 8 self-attention layers.


## Training

### Data

Google collected large-scale **real-world** demonstration data with 130K and 3000 demonstrations for training and evaluation respectively. The data collection time spent around 17 months.

A robot action consists of 7 dimensions $(x, y, z, \text{roll}, \text{pitch}, \text{yaw}, \text{opening of gripper})$ for a 7-DOF manipulator pose and 3 dimensions $(x, y, \text{yaw})$ for a base pose

Each action variable is equally *discretized* into 256 bins.

### Loss

Categorical cross-entropy loss

### Resources

*Not mentioned*

## Performance

RT-1 outperforms other baselines in seen tasks.

| <img src="https://github.com/trapoom555/trapoom555-blog/blob/main/static/images/RT1/performance.png?raw=true" style= "display: block; margin-left: auto; margin-right: auto; width: 100%;"/>|
|:--:| 
| *RT-1 and other baselines performance in **seen tasks** (Image from [[1]](#1))* |

To further test on generalization, the models are tested on 3 different settings levels

1. L1 : new counter-top layout & lightning condition
2. L2 : add unseen distractor objects
3. L3 : new task in unseen location

| <img src="https://github.com/trapoom555/trapoom555-blog/blob/main/static/images/RT1/performance_generalize.png?raw=true" style= "display: block; margin-left: auto; margin-right: auto; width: 100%;"/>|
|:--:| 
| *RT-1 has better generalization performance over other baselines (Image from [[1]](#1))* |

## Insights

The experiment shows that data diversity is more essential than data quantity.

| <img src="https://github.com/trapoom555/trapoom555-blog/blob/main/static/images/RT1/insight_data_size.png?raw=true" style= "display: block; margin-left: auto; margin-right: auto; width: 80%;"/>|
|:--:| 
| *RT-1 with different data sizes (Image from [[1]](#1))* |


Adding simulation data does not impact the performance on real objects, while significantly improving real performance on
objects that were only introduced in simulation (+64%). It also improves real-world generalization
on simulated objects used with skills seen only in the real world (+26%), e.g. “move X to Y” where
X only appeared in simulated “pick X” task.

| <img src="https://github.com/trapoom555/trapoom555-blog/blob/main/static/images/RT1/insight_sim.png?raw=true" style= "display: block; margin-left: auto; margin-right: auto; width: 100%;"/>|
|:--:| 
| *RT-1 with simulation data (Image from [[1]](#1))* |

## Discussions and Limitations

- Real-world data is expensive. If we experiment on a different robot (fixed manipulator for example), we need to recollect the data.
- Again, the action output is discretized similar to [VIMA](https://trapoom555.github.io/trapoom555-blog/posts/vima/) [[7]](#7) and [PerAct](https://trapoom555.github.io/trapoom555-blog/posts/peract/) [[8]](#8). This approach is not applicable to tasks which are sensitive to path.
- Imitation Learning Method cannot surpass the performance of the demonstrators
- RT-1 cannot generalize to completely new motion that has not been seen before
- The tasks are not dexterous
- I think the data size required for the model to make it successful is quite large. Seeing at the data size experiment above, by using 20% of the entire data or to be precise, 26K samples, can achieve success rate only 60% with low generalization ability.
- In my point of view, there're many trainable weights which makes the model requires more data.

## References

<a id="1">[1]</a> 
Brohan, Anthony, et al. "Rt-1: Robotics transformer for real-world control at scale." arXiv preprint arXiv:2212.06817 (2022).

<a id="2">[2]</a> 
Ethan Perez, Florian Strub, Harm de Vries, Vincent Dumoulin, and Aaron Courville. Film: Visual
reasoning with a general conditioning layer. Proceedings of the AAAI Conference on Artificial
Intelligence, 32(1), Apr. 2018. doi: 10.1609/aaai.v32i1.11671.

<a id="3">[3]</a> 
Mingxing Tan and Quoc Le. EfficientNet: Rethinking model scaling for convolutional neural networks.
In Kamalika Chaudhuri and Ruslan Salakhutdinov (eds.), Proceedings of the 36th International
Conference on Machine Learning, volume 97 of Proceedings of Machine Learning
Research, pp. 6105–6114. PMLR, 09–15 Jun 2019.

<a id="4">[4]</a> 
Daniel Cer, Yinfei Yang, Sheng-yi Kong, Nan Hua, Nicole Limtiaco, Rhomni St John, Noah Constant,
Mario Guajardo-Cespedes, Steve Yuan, Chris Tar, et al. Universal sentence encoder. arXiv
preprint arXiv:1803.11175, 2018.

<a id="5">[5]</a> 
Michael Ryoo, AJ Piergiovanni, Anurag Arnab, Mostafa Dehghani, and Anelia Angelova. Tokenlearner:
Adaptive space-time tokenization for videos. Advances in Neural Information Processing
Systems, 34:12786–12797, 2021.

<a id="6">[6]</a> 
Vaswani, Ashish, et al. "Attention is all you need." Advances in neural information processing systems 30 (2017).

<a id="7">[7]</a> 
Jiang, Yunfan, et al. "Vima: General robot manipulation with multimodal prompts." arXiv (2022).

<a id="8">[8]</a> 
Shridhar, Mohit, Lucas Manuelli, and Dieter Fox. "Perceiver-actor: A multi-task transformer for robotic manipulation." Conference on Robot Learning. PMLR, 2023.