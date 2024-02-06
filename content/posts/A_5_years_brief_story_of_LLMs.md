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

Unlike GPT1 which takes unidirectional signal to predict the next token, BERT (Bidirectional Encoder Representation from Transformer) [] consider *Bidirectional Signal* to make language understanding better. BERT outperformed other state-of-the-art in many language tasks. It can even outperform a task-specific model and GPT-1. It implies that Bidirectional encoding is a useful way to represent language. And using pre-training language model as a base model (similar to GPT1 idea) and fine-tuning it is a way to make predictions more accurate.

## Architecture

BERT is an *encoder only* Transformer model. $\text{BERT}_ \text{BASE}$ has 12 layers with 110M parameters. The size is similar to the GPT1 for comparison perposes. There's also a $\text{BERT}_ \text{LARGE}$ which comprises of 24 layers with the size of 340M parameters.

| <img src="https://github.com/trapoom555/trapoom555-blog/blob/main/static/images/A_5_years_brief_story_of_LLMs/BERT_arch.png?raw=true" style= "display: block; margin-left: auto; margin-right: auto; width: 100%;"/>|
|:--:| 
| *BERT Architecture (Image from [])* |

BERT consists of 2 training stages similar to GPT1. The first stage is pre-training and then fine-tuning to a specific downstream tasks.

## Pre-training

## Input

There'll be 2 sentences as a BERT's input to make it applicable to many downstream tasks. Two sentences are separated by a [SEP] token. In each sentence, there'll be [MASK] tokens at random positions. At the starting point of the input, [CLS] is added to be a room for the sentence-level output prediction "C".



| <img src="https://github.com/trapoom555/trapoom555-blog/blob/main/static/images/A_5_years_brief_story_of_LLMs/BERT_input_example.png?raw=true" style= "display: block; margin-left: auto; margin-right: auto; width: 60%;"/>|
|:--:| 
| *BERT Input Example (Image from [])* |

### Input Masking

$15\\%$ of the token positions are chosen to be
1. $80\\%$ replace with [MASK] tokens
2. $10\\%$ choose random tokens
3. $10\\%$ leave them as they are

Note that the $15\\%$ of the tokens are not always masked because in the fine-tuning stage, there's no [MASK] tokens.

### Input representation

The representation mainly follows Transformer. And also added Segment Embeddings to indicate whether the token is in the first sentence or the second one.

| <img src="https://github.com/trapoom555/trapoom555-blog/blob/main/static/images/A_5_years_brief_story_of_LLMs/BERT_input_representation.png?raw=true" style= "display: block; margin-left: auto; margin-right: auto; width: 100%;"/>|
|:--:| 
| *BERT Input Representation (Image from [])* |

## Objectives

Pre-training's goal is to achieve these 2 objectives.

### Masked LM

This objective is to map output tokens to  correct tokens while the inputs of that token are masked. 

It's quite similar to the GPT1 objective but the differences are only that the masked input is in the middle of the sentence and bidirectional signal is used to predict that token.

### Next Sentence Prediction (NSP)

In order to model the relationship between two sentences, the model will predict whether the second sentence in the input is a consecutive sentence to the first one. The prediction is a binary value which will output at the first token of the output sequence (denoted as "C").

## Downstream Tasks

Due to the flexible structure of BERT, it can be applied to many downstream tasks by end-to-end fine-tuning it.

Most of the time if the downstram tasks are to predict the sentence level-output, the first output token "C" is used for the prediction output. But if the output of the downstream tasks depend on an input sequence length, then the token corresponding to the input token will be the output of the downsteam tasks.

| <img src="https://github.com/trapoom555/trapoom555-blog/blob/main/static/images/A_5_years_brief_story_of_LLMs/BERT_downstream_tasks.png?raw=true" style= "display: block; margin-left: auto; margin-right: auto; width: 100%;"/>|
|:--:| 
| *BERT in Downstream Tasks (Image from [])* |

# GPT2 (2019.02)

# References