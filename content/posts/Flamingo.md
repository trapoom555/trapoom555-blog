---
author: "trapoom555"
title: "Flamingo : a Visual Language Model for Few-Shot Learning"
date: "2024-01-16"
description: "I'm a bird, just kidding, I'm a VLM 🦩 (Published: 2022.04, Deepmind)"
tags: [
    "multimodal",
    "computer-vision",
    "NLP",
    "few-shot",
    "zero-shot",
    "transfer-learning",
    visual-language-model,
    "paper-review"
]
math: mathjax
---

# Background

*Few-shot learning*, a.k.a. learning new tasks by a few examples, is quite simple for a human. But it has been a challenging task for machine learning for quite long time. In a typical visual models, it requires tens of thousands of example to train the model to adapt to the new task. 

Flamingo [[1]](#1) is a Visual Language Model (VLM) which can rapidly adapt to new tasks. It has a very strong *few-shot learning* capability.

# Flamingo 🦩

Flamingo provides a bridge between powerful vision model and language model. The Flamingo is very flexible. It can seamlessly ingest images, videos and text as input. And can handle with **arbitrarily interleaved sequence** of text and image.

## Architecture

| <img src="https://github.com/trapoom555/trapoom555-blog/blob/main/static/images/Flamingo/flamingo_architecture.png?raw=true" style= "display: block; margin-left: auto; margin-right: auto; width: 100%;"/>|
|:--:| 
| *Flamingo Architecture (Image from [[1]](#1))* |

### Visual Processing
- **Vision Encoder** <br>
A pre-trained NF-ResNet [[2]](#2) is used as a Vision Encoder. It's trained with the contrastive objective with image and text pairs. And it is frozen ❄️ in a Flamingo settings.

- **Perceiver Resampler** <br>
Perceiver Resampler [[3]](#3) recieves 1-D flatten spatio-temperal features from the Vision Encoder and output a **fixed number** of Visual Tokens (64) to reduce a computational complexity.

### Language Models 

Chinchilla [[4]](#4) with different sizes (1.4B, 7B and 70B) are adopted as Language Models. They will be combined to the Vision Encoder and other components in the model and will produce Flamingo 1.4B, 7B and 80B respectively.

> It is frozen ❄️ to prevent "Catastrophic Forgetting", the ablation study shows that fine-tuning the LLM layer dropped the performance by 8%

### Fusing Visual and Text information

| <img src="https://github.com/trapoom555/trapoom555-blog/blob/main/static/images/Flamingo/flamingo_xattn_dense.png?raw=true" style= "display: block; margin-left: auto; margin-right: auto; width: 100%;"/>|
|:--:| 
| *GATED XATTN-DENSE Layer (Image from [[1]](#1))* |

#### GATED XATTN-DENSE

**Cross Attention**

Cross Attention layers are added between the frozen pre-trained language model layers. The inner architecture is quite similar to the original [Transformer](https://trapoom555.github.io/trapoom555-blog/posts/transformer/) [[5]](#5) block, using Text information as queries and Visual information (connected from the Perceiver Resampler) as keys and values. 

**Tanh Gating**

The only difference to the original Transformer block is that there's a $\tanh(\alpha)$ gating [[6]](#6) multiplies to the output of a preceeding sublayer.

> It is done to ensure that at initialization, the conditioned model yields the same results as the original language model. $\alpha$ is learnable and set to $0$ at the initialization. It makes $\tanh(\alpha) = 0$. So that, at the beginning, the results of the conditioned model should be equal to the results from only LLM. By adding this gate **it improves the model's training stability and performance.**


## Data

Deepmind used various large-scale datasets of text, image or video pairs to train Flamingo. It's reported that using various dataset can boost the model performance.

- **M3W** : MultiModal MassiveWeb dataset (self-collected by Deepmind), 43 million webpages. The dataset was sampled to have a max token length of 256 tokens and take up to 5 images.
- **Image/Video and text pairs** : Deepmind leverage the ALIGN dataset [[7]](#7), 1.8 billion image-text pairs, by adding self-collected dataset which consists of 312 million image-text pairs and 26 million short videos (below 22 seconds) text pairs. Note that the videos are sampled at 1 FPS.

## Training

### Likelihood
Like other Large Language Models (LLMs), the model is trained in Self-supervised fasion, or in the other word, it tries to predict the next token correctly. Writing it down mathematically, the model is trying to maximize the following **likelihood**
    $$p(y|x) = \prod_{l=1}^L p(y_l|y_{<l}, x_{\le l})$$

Where $y$ is the texts conditioned with interleaved images and videos $x$, $y_l$ is the $l$-th token of text input, $y_{<l}$ is the set of preceeding language tokens, $x_{\le l}$ is the set of images/videos tokens preceeding $y_l$

### Objective
Minimizing a weighted sum of **per-dataset** expected negative log-likelihoods
$$\sum_{m=1}^{M} \lambda_m \cdot \mathbb E_{(x, y)\sim \mathcal D_m} \left[ - \sum_{l=1}^L \log p(y_l | y_{<l}, x_{\le l})\right]$$

Where $\lambda_m$ is a weight for each $\mathcal D_m$ dataset

### Per-image / Video Masking

In the cross attention layer, **just one previous image** is attended directly with the text tokens. The rest preceeding image tokens are masked. And still, the information of all previous images are kept in the self-attention layer. This helps the model to generalize the number of visual input in the sequence. (experimentally supported)



## Performance

The performance of Flamingo is stunning, with large-scale training data, it shows a strong in-context few-shot learning capabilities

1. It can outperform **all** SOTA zero-shot or few-shot methods on 16 benchmarks
2. Surpasses fine-tuned model on 6 out of 16 benchmarks. But keep in mind that Flamingo's capabilities are versatile. ([Frozen](https://trapoom555.github.io/trapoom555-blog/posts/frozen/) [[8]](#8) can't do this)

| <img src="https://github.com/trapoom555/trapoom555-blog/blob/main/static/images/Flamingo/flamingo_eval.png?raw=true" style= "display: block; margin-left: auto; margin-right: auto; width: 100%;"/>|
|:--:| 
| *Flamingo outperforms all SOTA zero-shot or few-shot methods on 16 benchmarks (Image from [[1]](#1))* |

## Limitations

- As it was built on LLM, it directly inherits their weekness
    - Hallucination
    - LLMs generalize poorly to sequences longer than the training ones
- The classification performance of Flamingo lags behind the SOTA of contrastive models
- In-context learning is sensitive and costly at inference which leads to **poor scalability**

# References
<a id="1">[1]</a> 
Alayrac, Jean-Baptiste, et al. "Flamingo: a visual language model for few-shot learning." Advances in Neural Information Processing Systems 35 (2022): 23716-23736.

<a id="2">[2]</a> 
Brock, Andy, et al. "High-performance large-scale image recognition without normalization." International Conference on Machine Learning. PMLR, 2021.

<a id="3">[3]</a> 
Andrew Jaegle, Felix Gimeno, Andy Brock, Oriol Vinyals, Andrew Zisserman, and Joao Carreira. Perceiver: General perception with iterative attention. In International Conference on Machine Learning, 2021.

<a id="4">[4]</a> 
Jordan Hoffmann, Sebastian Borgeaud, Arthur Mensch, Elena Buchatskaya, Trevor Cai, Eliza Rutherford, Diego de Las Casas, Lisa Anne Hendricks, Johannes Welbl, Aidan Clark, Eric Noland Tom Hennigan, Katie Millican, George van den Driessche, Bogdan Damoc, Aurelia Guy, Simon Osindero, Karen Simonyan, Erich Elsen, Jack W. Rae, Oriol Vinyals, and Laurent Sifre. Training compute-optimal large language models. arXiv:2203.15556, 2022.

<a id="5">[5]</a> 
Vaswani, Ashish, et al. "Attention is all you need." Advances in neural information processing systems 30 (2017).

<a id="6">[6]</a> 
Sepp Hochreiter and Jürgen Schmidhuber. Long short-term memory. Neural Computation, 1997.

<a id="7">[7]</a> 
Chao Jia, Yinfei Yang, Ye Xia, Yi-Ting Chen, Zarana Parekh, Hieu Pham, Quoc V. Le, Yun- Hsuan Sung, Zhen Li, and Tom Duerig. Scaling up visual and vision-language representation learning with noisy text supervision. arXiv:2102.05918, 2021.

<a id="8">[8]</a> 
Tsimpoukelli, Maria, et al. “Multimodal few-shot learning with frozen language models.” Advances in Neural Information Processing Systems 34 (2021): 200-212.