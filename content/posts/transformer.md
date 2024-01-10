---
author: "trapoom555"
title: "Transformer : The idea behind the AI revolution"
date: "2024-01-09"
description: "Understand the beginning of GPT, BERT, ViT, etc."
tags: [
    "NLP",
    "basic-AI-knowledge",
    "paper-review"
]
math: katex
draft: true
---

## Background

> Most of data we've experienced in everyday life is in *sequence*. 

Sequential data is the type of data which order matters to its meaning. The most common exmaple for sequential data is natural language. We understand characters in order as a word and words in order as a sentence or phrase. Other examples of the sequential data are time series data, audio signals, video streams, stock price.

### Sequence Modeling
is the task that we try to find a pattern of the sequential data. It can be used to extract the insight of the data and even be used to predict the future data in the sequence. It has been used in several applications, for example
- Time Series Forcasting
- Machine Translation
- Text Summarization
- Speech Recognition
- etc.

Before Transformer [[1]](#1), there're several research attempt to tackle sequence modeling problem.
One of the most successful model was the LSTM (Long Short-term Memory) [[2]](#2) which is a special kind of Recurrent Neural Network (RNN).

### Recurrent Neural Network

Recurrent Neural Network (RNN) is a type of Neural Network that enables sequential data processing capability of the Neural Network by passing through the *state* at all time steps. 

The architecture is quite simple. There's a RNN unit that receives an input vector at timestep $t$ as 
$ \bm x_t $ and the activation vector of the previous timestep $t-1$ as $\bm a_{t-1}$. It will then calculate the activation vector $\bm a_t$ for the next time step. And uses $\bm a_t$ to calculate the output vector $\bm y_t$ of the RNN at that time step.

| ![RNN architecture picture](https://github.com/trapoom555/trapoom555-page/blob/main/static/images/transformer/RNN.png?raw=true) |
|:--:| 
| *RNN Architecture (Image by author)* |


The RNN can be formulated as follows.

$$\bm a_t = f(\bm x_t, \bm a_{t-1})$$
$$\bm y_t = g(\bm a_t)$$


## References
<a id="1">[1]</a> 
Vaswani, Ashish, et al. "Attention is all you need." Advances in neural information processing systems 30 (2017).

<a id="2">[2]</a>
Sepp Hochreiter and Jürgen Schmidhuber. Long short-term memory. Neural computation, 9(8):1735–1780, 1997.