---
author: "trapoom555"
title: "Frozen : Transfer LLM knowledge to Multimodal"
date: "2024-01-15"
description: "It's also a Few-shot learner (Published: 2021.07, Deepmind)"
tags: [
    "multimodal",
    "computer-vision",
    "NLP",
    "few-shot",
    "zero-shot",
    "transfer-learning",
    "paper-review"
]
math: mathjax
draft: true
---

# Background

Large Language Models (LLMs) are *few-shot learners*. They can learn new tasks without any gradient updates. It is sometimes called *in-context learning*. But there's a limitation that LLM can understand just only text data. This paper [[1]](#1) extends the capability of *few-shot learners* to multimodal settings and proved that the knowledge from LLM can be transfer to the multimodal model built upon it.

# Frozen ❄️

## Architecture

With a pre-trained LLM having parameters $\theta$, It can be modified to have an understanding of image data by using an additional Neural Network called Vision Encoder ($v_\phi$) which can transform the image into  word token space. The image tokens will then be *prepended* to the text tokens before feeding into the LLM. The image tokens are named as a **visual prefix** since it's a prefix of the text tokens.

In this paper, NF-ResNet50 [[2]](#2) was used as a backbone of the Vision Encoder. And the output vector of NF-ResNet50 is linear transformed by learnable matrix in order to have the same dimension as text embeddings.

| <img src="https://github.com/trapoom555/trapoom555-blog/blob/main/static/images/Frozen/frozen_architecture.png?raw=true" style= "display: block; margin-left: auto; margin-right: auto; width: 80%;"/>|
|:--:| 
| *Frozen Architecture (Image from [[1]](#1))* |


## Mathematics Formulation

Suppose there's a sequence of $N$ images $\boldsymbol X_I \in \mathbb{R}^{N \times w\times h \times c}$, the Vision Encoder $v_\phi$ will transform the images to image embedding vectors $\boldsymbol O_I \in \mathbb{R}^{N \times E}$ which each vector has dimensionality $E$.

$$\boldsymbol O_I = v_\phi \left(\boldsymbol X_I \right)$$

To make the image and text embedding dimension the same, a *learnable* transformation matrix $W \in \mathbb{R}^{E \times D}$ is applied. And the image embedding vectors $\boldsymbol Y_I \in \mathbb{R}^{N\times D}$ are ready for the LLM.

$$\boldsymbol Y_I = \boldsymbol O_I \boldsymbol W $$

Speaking of the text embedding side, there's a pre-trained text embedder $g_\theta$ which gives us text embedding vectors as a matrix $\boldsymbol Y_T \in \mathbb{R}^{M\times D}$ where $M$ is the number of text tokens.

To formulate the proper input for the pre-trained [Transformer](https://trapoom555.github.io/trapoom555-blog/posts/transformer/) [[3]](#3) $f_\theta$, $\boldsymbol Y_I$ and $\boldsymbol Y_T$ are concatenated.

$$ \boldsymbol Y = \boldsymbol Y_I \oplus \boldsymbol Y_T$$


## Data

The Frozen was trained on 3 millon image-caption pairs from Conceptual Captions dataset [[4]](#4)

## Freezing the LLM weights

During training, the only weight updates happen at the Vision Encoder $v_\phi$ and the transformation matrix $\boldsymbol W$. **The weights in the pre-trained LLM are frozen.**

> The experiment shows that fine-tuning the LLM hurts the generalization since there're not many text-caption data compared to the text-only data. **The pre-trained LLM weights shold better be Frozen during training**

## Few-shot Learner
It was discovered that training a Frozen exhibits some few-shot learner capabilities.
1. Rapid adaption to new tasks
2. Fast access to general knowledge
3. Fast binding of visual and linguistic elements


# References
<a id="1">[1]</a>
Tsimpoukelli, Maria, et al. "Multimodal few-shot learning with frozen language models." Advances in Neural Information Processing Systems 34 (2021): 200-212.

<a id="2">[2]</a>
Brock, Andy, et al. "High-performance large-scale image recognition without normalization." International Conference on Machine Learning. PMLR, 2021.

<a id="3">[3]</a>
Vaswani, Ashish, et al. "Attention is all you need." Advances in neural information processing systems 30 (2017).

<a id="4">[4]</a>
Piyush Sharma, Nan Ding, Sebastian Goodman, and Radu Soricut. Conceptual captions: A cleaned, hypernymed, image alt-text dataset for automatic image captioning. In Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 2556–2565, 2018.