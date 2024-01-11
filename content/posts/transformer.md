---
author: "trapoom555"
title: "Transformer : The idea behind AI revolution"
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

Sequential data is the type of data which order matters to its meaning. The most common exmaple of sequential data is our language. We understand characters in order as a word and words in order as a phrase or sentence. Other examples of the sequential data are time series data, audio signals, video streams and stock price.

### Sequence Modeling
Sequence Modeling is the task that of finding *patterns* in the sequential data. It can be used to extract the insight of the data and even be used to predict the future data in the sequence. It has been used in several applications, for example
- Time Series Forcasting
- Machine Translation
- Text Summarization
- Speech Recognition
- etc.

Transformer [[1]](#1) is Neural Network architecture which is used in Sequence Modeling task. It was first used in a Machine Translation task and has completely revolutionized the AI realm.

Let's look at a bit of history. Before Transformer was proposed, there're several research attempt to tackle Sequence Modeling problem.
One of the most successful model was the LSTM (Long Short-term Memory) [[2]](#2) which is a special kind of Recurrent Neural Network (RNN).

### Recurrent Neural Network

Recurrent Neural Network (RNN) is a type of Neural Network that unlocks sequential data processing capability of the Neural Network by passing through the *state* at all time steps. 

The architecture is quite simple. There's a RNN unit that receives
- Input vector at timestep $t$ as 
$ \bm x_t $ 
- Hidden state of the previous timestep $t-1$ as $\bm h_{t-1}$. 

After some calculations, It will then output 

- Hidden state $\bm h_t$ for the next time step
- Output vector $\bm y_t$ of the RNN at time step $t$.

The RNN can be formulated as follows.

$$\bm h_t = f(\bm x_t, \bm h_{t-1})$$
$$\bm y_t = g(\bm h_t)$$

| ![RNN architecture picture](https://github.com/trapoom555/trapoom555-blog/blob/main/static/images/transformer/RNN.png?raw=true) |
|:--:| 
| *RNN Architecture (Image by author)* |

### RNN Limitations?

There're some important drawbacks in the RNN architecture

1. **The architecture can't be parallelized**

    The RNN processes input in sequence. It makes a training process very slow.

2. **Long-term dependencies problem**
    
    It's quite difficult for RNN to capture the information in long time ago. It make sense that keeping the information for a longer time needs more memory. But the number of dimensions of the hidden state vector is still the same. There's an information bottleneck at the hidden state vector.

Transformer can perfectly solve these problems !

## Transformer


### Overall Architecture

Transformer consists of $N$ identical Transformer blocks (In the original paper $N=6$). Each Transformer block consists of an encoder (left) and a decoder (right).

- Encoder (left) : encode the input information
- Decoder (right) : use encoded information along with the contextual input information to generate an output sequence

The Residual connection [[3]](#3) is applied to maintain an accuracy in a deep model. By utilizing the residual connection, all output vectors of each layer in the Transformer has to have the same dimension. In the original paper the dimension is set to be $d_{model}=512$

| <img src="https://github.com/trapoom555/trapoom555-blog/blob/main/static/images/transformer/transformer_architecture.png?raw=true" style= "display: block; margin-left: auto; margin-right: auto; width: 60%;"/>|
|:--:| 
| *Transformer Architecture (Image from [[1]](#1))* |

### Attention

### Multi-Head Attention

### Masked Multi-Head Attention

### Positional Embedding

### Input and Output of the Transformer

Since the Transformer was first used in Machine Translation. I'll give an example of the input and output in a format of Machine Translation task.


In the training phase, the whole sentences of `(source_language, target_language_with_right_shift)` will be fed into the model. 

| ![Transformer input output example](https://github.com/trapoom555/trapoom555-blog/blob/main/static/images/transformer/transformer_io_train.png?raw=true) |
|:--:| 
| *Transformer input output in traning phase (Image by author)* |


In the inference phase, the model will predict the next token in an auto-regressive manner.




## References
<a id="1">[1]</a> 
Vaswani, Ashish, et al. "Attention is all you need." Advances in neural information processing systems 30 (2017).

<a id="2">[2]</a>
Sepp Hochreiter and Jürgen Schmidhuber. Long short-term memory. Neural computation, 9(8):1735–1780, 1997.

<a id="3">[3]</a>
He, Kaiming, et al. "Deep residual learning for image recognition." Proceedings of the IEEE conference on computer vision and pattern recognition. 2016.