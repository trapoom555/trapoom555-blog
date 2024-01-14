---
author: "trapoom555"
title: "CLIP : Bridge between Image and text"
date: "2024-01-14"
description: "Learn image representation by text supervision. (Published: 2021)"
tags: [
    "multi-modal",
    "computer-vision",
    "NLP",
    "paper-review"
]
math: mathjax
---

# Background

There is a lot of research in computer vision field trying to find the State Of The Art (SOTA) architecture evaluating on ImageNet benchmark [[1]](#1). It turns out that many models perform well in the ImageNet benchmark, but giving a new data, the model's performance dropped significantly. Most of the time practitioners have to fine-tune the model to be able to apply SOTA model to their own data. This gives some doubts for the robustness of the model in computer vision tasks that can't be *zero-shot* to a new dataset.

And also, the model architecture which needs a "Machine Learning data format (i.e. output of the model needs to be one-hot encoding)", requires intensive and costly labors to build the training dataset. The previous approaches has low scalability.

| <img src="https://github.com/trapoom555/trapoom555-blog/blob/main/static/images/CLIP/clip_zero_shot.png?raw=true" style= "display: block; margin-left: auto; margin-right: auto; width: 100%;"/>|
|:--:| 
| *ResNET101 performs worse than CLIP in an unseen data proving that ResNET101 lacks of zero-shot capability to a new dataset (Image from [[2]](#2))* |

# CLIP

CLIP (Contrastive Language-Image Pre-training) [[3]](#3) has shown a potential on training with noisy large-scale data which is not in the "Machine Learning format" and can achieve a good *zero-shot* performance on a new dataset.

## Dataset

CLIP was trained on 400 million (text, image) pairs from the internet which is *much larger* than 14 millon samples in the ImageNet dataset.

## Architecture

It consists of 2 encoder backbones

1. Image Encoder ($\mathcal{I}$)

2. Text Encoder ($\mathcal{T}$)


In the paper, two different image encoders [Vision Transformer (ViT)](https://trapoom555.github.io/trapoom555-blog/posts/vit/) [[4]](#4) and ResNet [[5]](#5) (with modifications) were used in the experiment. And the text encoder adopted [Transformer](https://trapoom555.github.io/trapoom555-blog/posts/transformer/) [[6]](#6) (with modifications).

Suppose we have $\boldsymbol{X}_T$ as a batch of text input and $\boldsymbol{X}_I$ as a batch of image input. Both of them have the similar batch size of $N$. The text and image encoders will transform those inputs to output matrices $\boldsymbol{Y}_T \in \mathbb{R}^{N \times \text{text_embed}}$ and $\boldsymbol{Y}_I \in \mathbb{R}^{N \times \text{img_embed}}$ respectively.

$$\boldsymbol{Y}_T = \mathcal{T}(\boldsymbol{X}_T) \quad \boldsymbol{Y}_I = \mathcal{I}(\boldsymbol{X}_I)$$

The output vectors are then linearly transformed by *learnable* matrices in order to get the same embedding dimensions of text and image embeddings.

$$\boldsymbol{T} = \boldsymbol{Y}_T \boldsymbol{W}_T \quad \boldsymbol{I} = \boldsymbol{Y}_I \boldsymbol{W}_I$$

We now have the embedding vectors from text as $\boldsymbol{T} \in \mathbb{R}^{N\times e}$ and from image as $\boldsymbol{I} \in \mathbb{R}^{N\times e}$ where $e$ is the embedding dimensions.

## Training

CLIP jointly trains image and text encoder by using a *Contrastive Learning* concept.

> **Contrastive Learning** is the concept of learning embedding space which similar pairs are close to each other while dissimilar pairs are far apart.

By following the concept, CLIP designed training procedure and objectives as follows.

1. Calculate all possible cosine similarity pairs of text and image embedding vectors in a batch (Since we have $N$ samples in a batch, there'll be $N^2$ possible pairs)

2. Scale all cosine similarity pairs with temperature $\tau$

3. Constructing a loss function to be the Cross-Entropy loss. The objective is to maximize the cosine similarity between text and image embeddings of $N$ real pairs (blue entries in the image below) and minimize other wrong pairs.

$$\mathcal L_I = -\frac{1}{N}\sum_{i=1}^N \sum_{j=1}^N \mathbb{I}_{i,j} \log \left( \frac{\boldsymbol{I}_i \cdot \boldsymbol{T}_j}{ \Vert \boldsymbol{I}_i \Vert_2 \Vert \boldsymbol{T}_j \Vert_2} e^\tau \right)$$

$$\mathcal L_T = -\frac{1}{N}\sum_{i=1}^N \sum_{j=1}^N \mathbb{I}_{i,j} \log \left( \frac{\boldsymbol{T}_i \cdot \boldsymbol{I}_j}{\Vert \boldsymbol{T}_i \Vert_2 \Vert \boldsymbol{I}_j \Vert_2} e^\tau \right)$$

$$\mathbb{I}_{i,j} = \begin{cases} 1 \quad \text{if } i= j \\\\
0 \quad \text{otherwise}\end{cases}$$

> Since the cosine similarity has a symmetric property, $\mathcal L_I = \mathcal L_T$


| <img src="https://github.com/trapoom555/trapoom555-blog/blob/main/static/images/CLIP/clip_train.png?raw=true" style= "display: block; margin-left: auto; margin-right: auto; width: 80%;"/>|
|:--:| 
| *CLIP training procedure (Image from [[3]](#3))* |

## Zero-Shot

CLIP can achieve two *zero-shot* aspects

1. Can perform well on a new dataset <br>
CLIP can do this job better than ImageNet-trained models due to its large-scale training dataset.
2. Can be apply nearly arbitrary visual classification tasks <br>
Unlike ImageNet-trained models that can predict just pre-defined classes in the training dataset, CLIP can adapt to all labels<br>
    - Use the text encoder to calculate an embedding vectors of **all given labels** <br> Suppose $\boldsymbol X_T$ is a matrix of $N$ labels $$\boldsymbol T = \mathcal{T}(\boldsymbol X_T) \boldsymbol W_T$$
    - Pass an image through the image encoder $$\boldsymbol I_1 = \mathcal{I}(\boldsymbol x_I) \boldsymbol W_I$$
    - Calculate cosine similarity of an image to all label embeddings, scale by temperature and then normalize to a probability distribution with the softmax function $$p(\boldsymbol X_{T,i} | \boldsymbol x_I) = \frac{\exp \left(\boldsymbol (\boldsymbol T_i \cdot \boldsymbol I_1) \tau \right)}{\sum_{i=1}^N \exp \left(\boldsymbol (\boldsymbol T_i \cdot \boldsymbol I_1) \tau \right)}$$


| <img src="https://github.com/trapoom555/trapoom555-blog/blob/main/static/images/CLIP/clip_zero_shot_pred.png?raw=true" style= "display: block; margin-left: auto; margin-right: auto; width: 80%;"/>|
|:--:| 
| *CLIP zero-shot in the image classification task (Image from [[3]](#3))* |

## Limitations

- CLIP fails to do more complex task than recognizing common object. For example, counting people in the image. 
- It sensitives to wordings (need Prompt Engineering)
- CLIP also still has poor generalization to images not covered in its pre-training dataset. But still better than the models that were trained on the ImageNet dataset because CLIP was trained on a much larger dataset.

| <img src="https://github.com/trapoom555/trapoom555-blog/blob/main/static/images/CLIP/clip_prompt_engineering.png?raw=true" style= "display: block; margin-left: auto; margin-right: auto; width: 70%;"/>|
|:--:| 
| *Prompt Engineering affects outcomes in CLIP (Image from [[3]](#3))* |

# References
<a id="1">[1]</a> 
J. Deng, W. Dong, R. Socher, L. -J. Li, Kai Li and Li Fei-Fei, "ImageNet: A large-scale hierarchical image database," 2009 IEEE Conference on Computer Vision and Pattern Recognition, Miami, FL, USA, 2009, pp. 248-255, doi: 10.1109/CVPR.2009.5206848.

<a id="2">[2]</a> 
OpenAI. (n.d.). Clip: Connecting text and images. CLIP: Connecting text and images. https://openai.com/research/clip#fn-2 

<a id="3">[3]</a> 
Radford, Alec, et al. "Learning transferable visual models from natural language supervision." International conference on machine learning. PMLR, 2021.

<a id="4">[4]</a> 
Dosovitskiy, Alexey, et al. "An image is worth 16x16 words: Transformers for image recognition at scale." arXiv preprint arXiv:2010.11929 (2020).

<a id="5">[5]</a>
He, Kaiming, et al. "Deep residual learning for image recognition." Proceedings of the IEEE conference on computer vision and pattern recognition. 2016.

<a id="6">[6]</a> 
Vaswani, Ashish, et al. "Attention is all you need." Advances in neural information processing systems 30 (2017).
