---
author: "trapoom555"
title: "Transformer : The idea behind AI revolution"
date: "2024-01-12"
description: "Foundation of BERT, GPT, ViT, etc."
tags: [
    "NLP",
    "basic-AI-knowledge",
    "paper-review"
]
math: mathjax
---

# Background

> Most of data we've experienced in everyday life is in *sequence*. 

Sequential data is a type of data which order matters to its meaning. The most common exmaple of the sequential data is our language. We understand characters in order as a word and words in order as a phrase or sentence. Other examples of the sequential data are time series data, audio signals, video streams and stock price data.

## Sequence Modeling
Sequence Modeling is the task that of finding *patterns* in the sequential data. It can be used to extract the insight of the data and also to predict the future samples in a sequence. It has been used in several applications, for example
- Time Series Forcasting
- Machine Translation
- Text Summarization
- Speech Recognition
- etc.

Transformer [[1]](#1) is a type Neural Network architecture which is used in Sequence Modeling task. It was first used in a Machine Translation task and has completely revolutionized the AI realm.

Before Transformer was proposed, there're several research attempt to tackle Sequence Modeling problem.
One of the most successful model was the LSTM (Long Short-term Memory) [[2]](#2) which is a special kind of Recurrent Neural Network (RNN).

## Recurrent Neural Network

Recurrent Neural Network (RNN) is a type of Neural Network which is used in a sequence modeling task. It can model the sequence by passing through the *state* at all time steps. 

The architecture is quite simple. There's a RNN unit that receives
- Input vector at timestep $t$ as 
$ \boldsymbol x_t $ 
- Hidden state of the previous timestep $t-1$ as $\boldsymbol h_{t-1}$

After some calculations, It will then output 

- Hidden state $\boldsymbol h_t$ for the next timestep
- Output vector $\boldsymbol y_t$ of the RNN at timestep $t$

The RNN can be formulated as follows.

$$\boldsymbol h_t = f(\boldsymbol x_t, \boldsymbol h_{t-1})$$
$$\boldsymbol y_t = g(\boldsymbol h_t)$$

| ![RNN architecture picture](https://github.com/trapoom555/trapoom555-blog/blob/main/static/images/transformer/RNN.png?raw=true) |
|:--:| 
| *RNN Architecture (Image by author)* |

## RNN Limitations

There're some important drawbacks in the RNN architecture

1. **The architecture can't be parallelized in training**

    The RNN processes input in sequence. It makes a training process very slow.

2. **Long-term dependencies problem**
    
    It's quite difficult for RNN to capture the information at long time ago. It makes sense that keeping the information for a longer time needs more memory. But the dimensions of the hidden state vector stays the same. *There's an information bottleneck at the hidden state vector.*

Transformer can perfectly solve these problems !

# Transformer


## Overall Architecture

Transformer consists of $N$ identical Transformer blocks (In the original paper $N=6$). Each Transformer block consists of an encoder (left) and a decoder (right).

- Encoder (left) : encode the input information
- Decoder (right) : use encoded information along with the contextual input information to generate an output sequence

The Residual connection [[3]](#3) is applied to maintain an accuracy in a deep model. By utilizing the residual connection, all dimension of output vectors at each layer in the Transformer has to be the same. In the original paper the dimension is set to be $d_{model}=512$

| <img src="https://github.com/trapoom555/trapoom555-blog/blob/main/static/images/transformer/transformer_architecture.png?raw=true" style= "display: block; margin-left: auto; margin-right: auto; width: 60%;"/>|
|:--:| 
| *Transformer Architecture (Image from [[1]](#1))* |

## Attention Mechanism

The Attention Mechanism plays an important role in the Transformer architecture. In fact, this idea was proposed long time ago. The idea of Attention can be easily understandable by this analogy,

> Human's eyes see the world in an interesting way. We usually focus on something interesting instead of looking at the entire scene at once

Like in a human, when focusing on something more meaningful, the AI can perform better. The Attention Mechanism has beed discovered to be useful in a field of image recognition, machine translation. In the Transformer architecture, Scaled-dot product attention, a special type of the Attention Mechanism, is applied.

The image below illustrates an Attention in a sentence of the word "making". Seeing that the word "making" has strong attention with "more" and "difficult" which complete the meaningful phrase "making...more difficult".

| ![Transformer input output example](https://github.com/trapoom555/trapoom555-blog/blob/main/static/images/transformer/attention_example.png?raw=true) |
|:--:| 
| *Attention for the word "making" with its own sentence [Self-Attention] (Image from [[1]](#1))* |

### Scaled-dot product Attention

Scaled-dot product is a special type of an Attention Mechanism which is used in the Transformer architecture. There're 2 types separating by their inputs.

1. **Self-Attention** : Calculate attentions within its the sequence. <br>
the input vectors $\boldsymbol x \in \mathbb{R}^{d_{model}}$ of each word, when coming into the Self-Attention layer, are packed into a matrix $\boldsymbol X \in \mathbb{R}^{\text{seq\_len} \times d_{model}}$ and will be then transformed to Query $\boldsymbol Q$, Key $\boldsymbol K$ and Value $ \boldsymbol V$ by these following equations.
$$\boldsymbol Q = \boldsymbol X \boldsymbol W^Q \quad \boldsymbol K = \boldsymbol X \boldsymbol W^K \quad \boldsymbol V = \boldsymbol X \boldsymbol W^V$$
$\boldsymbol Q, \boldsymbol K, \boldsymbol V \in \mathbb{R}^{\text{seq\_len} \times d_{model}}$<br> 
$\boldsymbol W^Q, \boldsymbol W^K, \boldsymbol W^V \in \mathbb{R}^{d_{model} \times d_{model}}$ are learnable transformation matrices.

2. **Encoder-Decoder Attention** : Calculate attentions across the sequences.<br>
Calculating $\boldsymbol Q$, $\boldsymbol K$ and $ \boldsymbol V$ in the Encoder-Decoder Attention is different from Self-Attention. The query $\boldsymbol Q$ comes from an encoder but $\boldsymbol K$ and $ \boldsymbol V$ come from a decoder. Suppose we have an input from an encoder as $\boldsymbol X \in \mathbb{R}^{\text{seq\_len} \times d_{model}}$ and input from a decoder as $\boldsymbol Y \in \mathbb{R}^{\text{seq\_len} \times d_{model}}$, $\boldsymbol Q$, $\boldsymbol K$ and $ \boldsymbol V$ can be calculated as follows
$$\boldsymbol Q = \boldsymbol X \boldsymbol W^Q \quad \boldsymbol K = \boldsymbol Y \boldsymbol W^K \quad \boldsymbol V = \boldsymbol Y \boldsymbol W^V$$

After $\boldsymbol Q$, $\boldsymbol K$ and $ \boldsymbol V$ have been calculated, A Scale-dot product attention can be computed as follows.

$$\text{Attention}(\boldsymbol Q, \boldsymbol K, \boldsymbol V)=\text{softmax}\left( \frac{\boldsymbol Q \boldsymbol K^T}{\sqrt{d_{k}}}\right) \boldsymbol V$$

In this case (number of heads is 1), $d_k =d_{model}$.

Let's breakdown each step of calculation to get some insights

1. $\boldsymbol Q \boldsymbol K^T \in \mathbb{R}^{\text{seq\_len} \times \text{seq\_len}}$ : This step perform a dot product operation which extracts the **similarities** of each pair between Queries and Keys (they can be also called the Attention Scores)

2. $\frac{\boldsymbol Q \boldsymbol K^T}{\sqrt{d_{k}}}$ : The calculated attention scores are reduced by $\sqrt{d_{k}}$ times. It prevents too large or too small scores that will fall into a very small gradient region of softmax which can cause a very slow gradient update.
| <img src="https://github.com/trapoom555/trapoom555-blog/blob/main/static/images/transformer/gradient_small.png?raw=true" style= "display: block; margin-left: auto; margin-right: auto; width: 60%;"/>|
|:--:| 
| *When input of the softmax function is too small or too large,<br> it will fall into a very small gradient region (Image by author)* |

3. $\text{softmax} \left( \frac{\boldsymbol Q \boldsymbol K^T}{\sqrt{d_{k}}}\right)$ : Translate scores to probabilities by applying softmax along the row axis. Now, each row will sum up to $1$.

4. $\text{softmax} \left( \frac{\boldsymbol Q \boldsymbol K^T}{\sqrt{d_{k}}}\right) \boldsymbol V \in \mathbb{R}^{\text{seq\_len} \times d_{model}}$ : Use the probabilities as weights in the weighted sum of values

> **Intuition** : the output of the Scale-dot product attention at row $i^{th}$ can be seen as a weighted linear combination of the value vectors where weights are the attentions of the $i^{th}$ query to all keys $$ \text{Attention}(\boldsymbol Q, \boldsymbol K, \boldsymbol V)_\text{i-th row} = \sum_j^\text{seq\_len} {\text{att}(\boldsymbol Q_i, \boldsymbol K_j) \boldsymbol V_j} $$
Meaning that, in the Machine Translation task, we have a *query* word in a source language and then we look at all *keys* in a target language, calculate an attention score and use it as weights to do a weighted sum with *values* in the target language. By doing this, the value associating with high attention will have more influence in output.


### Multi-Head Attention



We can further boost the performance of the Scale-dot product Attention by using multiple attention heads (or units).

> **Intuition** : If we have a word "Dad" as a query and "Son" as a key, do you think these two words have a high or low attention score ? The answer is "Could be high or low if we look at them in different aspects". "Dad" and "Son" could be similar if we consider that they are in a family member. But they could be different if we consider at their age. Using multiple attention heads **allow the model capture multiple aspects of attentions**.

Speaking mathematically, **it allows attention to learn different representation from different subspace**.

| <img src="https://github.com/trapoom555/trapoom555-blog/blob/main/static/images/transformer/multihead_attention.png?raw=true" style= "display: block; margin-left: auto; margin-right: auto; width: 40%;"/>|
|:--:| 
| *Multi-Head Attention (Image from [[1]](#1))* |

In the Multi-head Attention, $d_k = d_{model}/ h$ where $h$ is the number of heads and $d_k$ is the dimensions of embedding vectors in each head. In the paper, $h=8$ therefore, $d_k=512/8=64$.

We can compute the Multi-Head Attention as below.

$$\text{Multihead}(\boldsymbol Q, \boldsymbol K, \boldsymbol V) = \text{Concat}\left( \text{head}_1, ..., \text{head}_h\right) \boldsymbol W^O$$
$$\text{where head}_i= 
\begin{cases} 
 \text{Attention}\left( \boldsymbol X \boldsymbol W^Q, \boldsymbol X \boldsymbol W^K, \boldsymbol X \boldsymbol W^V\right) & \text{if Self-Attention } \\\\
 \text{Attention}\left( \boldsymbol X \boldsymbol W^Q, \boldsymbol Y \boldsymbol W^K, \boldsymbol Y \boldsymbol W^V\right) & \text{if Encoder-Decoder Attention}
\end{cases}$$

Note that 

$$\boldsymbol W^Q, \boldsymbol W^K, \boldsymbol W^V \in \mathbb{R}^{d_{model} \times d_k} \quad \text{head}_i \in \mathbb{R}^{\text{seq\_len} \times d_k}$$

$$\quad \boldsymbol W^O \in \mathbb{R}^{d_{model} \times d_{model}} \quad \text{Multihead}(\boldsymbol Q, \boldsymbol K, \boldsymbol V) \in \mathbb{R}^{\text{seq\_len} \times d_{model}}$$

### Masked Multi-Head Attention

Masked Multi-head Attention is used in the Decoder to ensure that the prediction of i-th token only depends on itself and the tokens before it. We do this because we want to maintain the autoregressive property. It can be formulated as below.

$$
\text{Masked-Attention}(\boldsymbol Q, \boldsymbol K, \boldsymbol V) =\text{softmax} \left( \frac{\boldsymbol Q \boldsymbol K^T}{\sqrt d_k}+ \boldsymbol M\right) \boldsymbol V
$$

The only difference from the Scale-dot product attention is that there's a Mask matrix $\boldsymbol M$ which is defined by

$$
\boldsymbol M_{i,j} = \begin{cases} -\infty & \text{if } i<j \\\\ 0 & \text{if } i=j \end{cases}
$$

To give more concrete example, Suppose we have scaled attention scores as follows

$$
\frac{\boldsymbol Q \boldsymbol K^T}{\sqrt d_k} = \begin{pmatrix}
a_{1,1} & a_{1,2} & a_{1,3} \\\\
a_{2,1} & a_{2,2} & a_{2,2} \\\\
a_{3,1} & a_{3,2} & a_{3,3} \\\\
\end{pmatrix} 
$$

After Adding a Mask, it will change upper triangular elements in a matrix to $-\infty$

$$
\frac{\boldsymbol Q \boldsymbol K^T}{\sqrt d_k} + \boldsymbol M = \begin{pmatrix}
a_{1,1} & -\infty & -\infty \\\\
a_{2,1} & a_{2,2} & -\infty \\\\
a_{3,1} & a_{3,2} & a_{3,3} \\\\
\end{pmatrix} 
$$

Applying the softmax makes the $-\infty$ terms come to $0$

$$ 
\text{Masked-Attention}(\boldsymbol Q, \boldsymbol K, \boldsymbol V) = \begin{pmatrix}
b_{1,1} & 0 & 0 \\\\
b_{2,1} & b_{2,2} & 0 \\\\
b_{3,1} & b_{3,2} & b_{3,3} \\\\
\end{pmatrix} \boldsymbol V
$$

> Seeing that the Masked-Attention score of a token i-th (at row $i$) depends on only the value of tokens at i-th and before it.




## Positional Encoding

Until now, the model has no clue about the position of each token. The job of the Positional Encoding is to add the *order* information of tokens to the input before feeding it into the Transformer. In math, it's just simply added to the input.

$$\boldsymbol X = \boldsymbol X + PE$$

There are learnable and fixed Positional Encoding. In this work, the fixed one was used. It's called the sine and cosine function with different frequencies.

$$PE_{(pos, 2i)} = sin\left( pos/10000^{2i/d_{model}}\right)$$
$$PE_{(pos, 2i+1)} = cos\left( pos/10000^{2i/d_{model}}\right)$$

Where $pos$ is the position of a token in a sequence and $i$ associates with dimension in embedding vector. The visualization of the Positional Encoding is below.


| <img src="https://github.com/trapoom555/trapoom555-blog/blob/main/static/images/transformer/positional_encoding.png?raw=true" style= "display: block; margin-left: auto; margin-right: auto; width: 100%;"/>|
|:--:|
| *Positional Encoding Visualization (Image by author)* |


## Feed-Forward Network

A feed-forward network (FFN) is applied after the Self-attention layers in each encoder and decoder. It consists of two linear transformation layers and a RELU as a nonlinear activation function within them, and can be denoted as the following function

$$ \text{FFN}(\boldsymbol X) = \boldsymbol W_2 \text{ReLU} (\boldsymbol W_1 \boldsymbol X + \boldsymbol b_1) + \boldsymbol b_2$$

## Input and Output of the Transformer

In this blog, I'll give an example in a Machine Translation task translating English to Thai.

### Training Phase

In the training phase, the Transformer will be given a whole sentence of the source language and a *right-shifted* target language. The objective is to predict the whole target language sentence (without shifting the position).

| <img src="https://github.com/trapoom555/trapoom555-blog/blob/main/static/images/transformer/transformer_io_train.png?raw=true" style= "display: block; margin-left: auto; margin-right: auto; width: 70%;"/>|
|:--:|
| *Transformer input output in traning phase (Image by author)* |

If you compare it with the training process in the RNN, the RNN sequentially predicts the target word (not a whole sentense like in the Transformer). Therefore, the training process of the Transformer is considered to be parallelized.


### Inference Phase

In the inference phase, the model will predict the next token in an autoregressive manner. Meaning that the outputs of the previous timestep will be fed to the model as the input at this timestep.

| <img src="https://github.com/trapoom555/trapoom555-blog/blob/main/static/images/transformer/transformer_io_infer.png?raw=true" style= "display: block; margin-left: auto; margin-right: auto; width: 70%;"/>|
|:--:|
| *Transformer input output in traning phase (Image by author)* |


# References
<a id="1">[1]</a> 
Vaswani, Ashish, et al. "Attention is all you need." Advances in neural information processing systems 30 (2017).

<a id="2">[2]</a>
Sepp Hochreiter and Jürgen Schmidhuber. Long short-term memory. Neural computation, 9(8):1735–1780, 1997.

<a id="3">[3]</a>
He, Kaiming, et al. "Deep residual learning for image recognition." Proceedings of the IEEE conference on computer vision and pattern recognition. 2016.