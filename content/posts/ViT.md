---
author: "trapoom555"
title: "ViT : Vision Transformer"
date: "2024-01-13"
description: "The Transformer does not limited to texts."
tags: [
    "computer-vision",
    "paper-review"
]
math: mathjax
---
# Background

Convolutional Neural Network (CNN) [[1]](#1) has been proven to be successful in many computer vision tasks for many years. And in the recent year (2017), [Transformer](https://trapoom555.github.io/trapoom555-blog/posts/transformer/) [[2]](#2) has achieved significant improvement in NLP tasks. Thinking out of the box, Vision Transformer (ViT) [[3]](#3) applied BERT [[4]](#4), a Transformer's variant, to a well-known computer vision task, image classification, directly and surpassed the CNN performance with large amount of training data.

# Vision Transformer (ViT)

| ![ViT architecture picture](https://github.com/trapoom555/trapoom555-blog/blob/main/static/images/ViT/ViT_architecture.png?raw=true) |
|:--:| 
| *ViT Architecture (Image from [[3]](#3))* |

## An Image as Tokens

The ViT recieves input as tokens because its Transformer backbone was used in NLP tasks before. The tokens are basically multiple 1-D embedding vectors. But the dimension of an image data is in 2-D (or 3-D if we count the RGB channels), so we need to have some input transformation to feed image into the ViT. This paper proposed the successful way of transforming an image to tokens.

Suppose we have an image $\boldsymbol X \in \mathbb{R}^{w \times h\times c}$ where $w$ is the image width, $h$ is the image height and $c$ is the image channels. The image is first patched into $n$ square pathes with the width and height of each patch as $p$. And then, each patch will be flattened. Therefore, we have an input for the ViT as $\boldsymbol X_p \in \mathbb{R}^{n \times (p^2 c)}$. $p=16$ pixels was used in the original paper.

## Prepending a [class] Token

Similar to BERT, a **learnable** token `[class]` is prepended. This token was introduced to be a room for aggregating information of all image patches at the last layer. In the end, the first token in the last layer will store the whole image representaion information.

## Other detils
- A **learnable** positional encoding is added to each token before going into the BERT model.
- Multi-Layer Perceptron (MLP) connects to the first token at the last layer to use a whole image representation information to do a pridiction task.




# ViT vs. CNN Insights

ViT needs very large amount of training data (about 14M-300M images) [[3]](#3) to perform better than the CNN. 

Because CNN introduces inductive biases within its structure which are
1. Shift-invariant : The locatin of objects in the image doesn't matter to the outcome.
2. Locality : Assumes that closer pixels are more related than further ones.

These biases are beneficial for extracting data from images. But it is proven that, without these biases and feeding more data, ViT surpasses CNN-based model in performance.


# Just for the Image Classification ?

After the discovery of the ViT in 2020, Vision Transformer has dominated and applied to many computer vision tasks.

| <img src="https://github.com/trapoom555/trapoom555-blog/blob/main/static/images/ViT/vison_transformer_tasks.png?raw=true" style= "display: block; margin-left: auto; margin-right: auto; width: 100%;"/>|
|:--:| 
| *Vision Transformer has been applied to many computer vision tasks and gained a good performance comparable to or surpass CNN-based architecture (Image from [[5]](#5))* |

# References

<a id="1">[1]</a> Y. LeCun et al. Gradient-based learning applied to document recogni- tion. Proceedings of the IEEE, 86(11):2278–2324, 1998.

<a id="2">[2]</a> 
Vaswani, Ashish, et al. "Attention is all you need." Advances in neural information processing systems 30 (2017).

<a id="3">[3]</a>
Dosovitskiy, Alexey, et al. "An image is worth 16x16 words: Transformers for image recognition at scale." arXiv preprint arXiv:2010.11929 (2020).

<a id="4">[4]</a>
Devlin, Jacob, et al. "Bert: Pre-training of deep bidirectional transformers for language understanding." arXiv preprint arXiv:1810.04805 (2018).

<a id="5">[5]</a>
Han, Kai, et al. "A survey on vision transformer." IEEE transactions on pattern analysis and machine intelligence 45.1 (2022): 87-110.