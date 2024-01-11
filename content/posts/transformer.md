---
author: "trapoom555"
title: "Transformer : The idea behind AI revolution"
date: "2024-01-09"
description: "Fundamental of BERT, GPT, ViT, etc."
tags: [
    "NLP",
    "basic-AI-knowledge",
    "paper-review"
]
math: mathjax
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
$ \boldsymbol x_t $ 
- Hidden state of the previous timestep $t-1$ as $\boldsymbol h_{t-1}$. 

After some calculations, It will then output 

- Hidden state $\boldsymbol h_t$ for the next time step
- Output vector $\boldsymbol y_t$ of the RNN at time step $t$.

The RNN can be formulated as follows.

$$\boldsymbol h_t = f(\boldsymbol x_t, \boldsymbol h_{t-1})$$
$$\boldsymbol y_t = g(\boldsymbol h_t)$$

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

### Attention Mechanism

Human's eyes see the world in an interesting way. We usually focus on something interesting instead of looking at the entire scene at once. This idea has been firstly adapted to AI for long time ago. it is called the Attention Mechanism. It has beed discovered to be useful in a field of image recognition, machine translation. A Scaled-dot product attention, a special type of the Attention Mechanism, is used in Transformer.

| ![Transformer input output example](https://github.com/trapoom555/trapoom555-blog/blob/main/static/images/transformer/attention_example.png?raw=true) |
|:--:| 
| *Attention for the word "making" with its own sentence [Self-Attention] (Image from [[1]](#1))* |

In the image above, Seeing that the word "making" has a strong attention with "more" and "difficult" which complete the phrase "making...more difficult"

#### Scaled-dot product Attention

When coming into the Self-Attention layer, the input vectors $\boldsymbol x \in \mathbb{R}^{d_{model}}$ of each word are packed into a matrix $\boldsymbol X \in \mathbb{R}^{\text{seq\_len} \times d_{model}}$ and will be then transformed to Query $\boldsymbol Q$, Key $\boldsymbol K$ and Value $ \boldsymbol V$.

$$\boldsymbol Q = \boldsymbol X \boldsymbol W^Q $$
$$\boldsymbol K = \boldsymbol X \boldsymbol W^K $$
$$\boldsymbol V = \boldsymbol X \boldsymbol W^V$$

Note that $\boldsymbol Q, \boldsymbol K, \boldsymbol V \in \mathbb{R}^{\text{seq\_len} \times d_{model}}$ and 
$\boldsymbol W^Q, \boldsymbol W^K, \boldsymbol W^V \in \mathbb{R}^{d_{model} \times d_{model}}$ are learnable transformation matrices.

A Scale-dot product attention can be calculated from $\boldsymbol Q, \boldsymbol K, \boldsymbol V$ as follows.

$$\text{Attention}(\boldsymbol Q, \boldsymbol K, \boldsymbol V)=\text{softmax}\left( \frac{\boldsymbol Q \boldsymbol K^T}{\sqrt{d_{model}}}\right) \boldsymbol V$$

Let's breakdown each step of calculation

1. $\boldsymbol Q \boldsymbol K^T \in \mathbb{R}^{\text{seq\_len} \times \text{seq\_len}}$ : This step perform a dot product operation which extracts the **similarity** pairs between Queries and Keys

2. $\frac{\boldsymbol Q \boldsymbol K^T}{\sqrt{d_{model}}}$ : The calculated similarity values are reduced by $\sqrt{d_{model}}$ times preventing too large or too small value that fall into a very small gradient region of softmax. It will cause a very small & slow gradient update.
| <img src="https://github.com/trapoom555/trapoom555-blog/blob/main/static/images/transformer/gradient_small.png?raw=true" style= "display: block; margin-left: auto; margin-right: auto; width: 60%;"/>|
|:--:| 
| *When input of the softmax function is too small or too large, it will fall into a very small gradient region (Image by author)* |

3. $\text{softmax} \left( \frac{\boldsymbol Q \boldsymbol K^T}{\sqrt{d_{model}}}\right)$ : Translate scores to probabilities by applying softmax along the row axis. It makes each row sum up to $1$.

4. $\text{softmax} \left( \frac{\boldsymbol Q \boldsymbol K^T}{\sqrt{d_{model}}}\right) \boldsymbol V \in \mathbb{R}^{\text{seq\_len} \times d_{model}}$ : Weighted sum of values

#### Multi-Head Attention



#### Masked Multi-Head Attention

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