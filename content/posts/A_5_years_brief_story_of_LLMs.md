---
author: "trapoom555"
title: "A 5 years brief story of LLMs (Large Language Models)"
date: "2024-02-05"
tags: [
    "LLM",
    "NLP",
    "paper-review",
    "series"
]
draft: true
math: mathjax
---

The discovery of a Transformer [] architecture is an AI revolution, many researchers has seen a potential of it and depelop research works on top of it. There're several famous models in this development lines including ChatGPT [] and Llama [] which we will dive into in this article.

# GPT 1 (2018.06)

GPT 1 (117 M parameters) [] is a very first LLM model which uses a benefit of semi-supervised pre-training stage. As its name, Generative Pre-trained Transformer, it is pre-trained in a very large corpus and then fine-tuned on downstream tasks. By doing semi-supervised pre-training, it doesn't need much effort in labeling. And there're a bunch of data which the model can use to train itself on. It's shown that using the large-scale data in a pre-training stage can make downstream tasks more accurate by transfering the knowledge from a pre-training stage.

## Architecture

The architecture is a 12-layer *decoder only* Transformer with causal masked self-attention heads. The causal masked self-attention heads don't allow the current token to depend on the next tokens.

| <img src="https://github.com/trapoom555/trapoom555-blog/blob/main/static/images/A_5_years_brief_story_of_LLMs/GPT1_arch.png?raw=true" style= "display: block; margin-left: auto; margin-right: auto; width: 30%;"/>|
|:--:| 
| *GPT-1 architecture is a 12-layer decoder only Transformer style blocks (Image from [[1]](#1))* |

On the top of the diagram, there're 2 green blocks which refer to the objective of pre-training and fine-tuning from left to right respectively.

## Pre-training Stage

It's done in a semi-supervised fashion, the objective is similar to language modeling which is to **predict the next token.**

## Fine-tuning Stage

Given a label $y$ and the input tokens $x^1, ..., x^m$, the model tries to maximize the following likelihood.

$$P(y\mid x^1, ..., x^m) = \text{softmax}(h_l^m W_y)$$

where $h_l^m$ is the sast transformer block's activation and $W_y$ is an learnable transformation matrix which maps activation space to label space.

## Apply to downstream tasks

| <img src="https://github.com/trapoom555/trapoom555-blog/blob/main/static/images/A_5_years_brief_story_of_LLMs/GPT1_downstream_tasks.png?raw=true" style= "display: block; margin-left: auto; margin-right: auto; width: 100%;"/>|
|:--:| 
| *GPT-1 can apply to many downstream tasks by adjusting a few details in the last layer (Image from [[1]](#1))* |

# BERT (2018.10)



# References