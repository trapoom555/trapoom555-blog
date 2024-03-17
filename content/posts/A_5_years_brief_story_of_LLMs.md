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

It's the first time the Language Model size hits 1B parameters. GPT2 [] is almost similar to GPT1 but moving Layer Normalization [] to the input of each sub-block. There are 4 sizes of the models in this paper. The largest one called GPT2 has a size of 1.5B parameters.

The key idea of this paper is to show that **A large enough Language Model can achieve zero-shot results without  fine-tuning.** This is the first idea of scaling a model's size to improve performances for the Language Models. The benefit of this discovered property is that we don't need any hand-craft datasets and also we don't need a specific model for the task, just scale the model and feed all data we have then the model should work on many tasks.

Since the Language model can perform on multi-tasks without fine-tuning, the task instructions are directly in the prompt giving the output conditioned by input and task.

$$p(\text{output} \mid \text{input}) \rightarrow p(\text{output} \mid \text{input, task})$$

# T5 (2019.10)

The idea of using one model for multi-tasks has become dominated. T5 [] (Text-To-Text Transfer Transformer) paper performed large emperical studies on several factors of building a model including input design, size, model archtecture and etc. to build a powerful framework, T5,  which can perform many language tasks in one model.

| <img src="https://github.com/trapoom555/trapoom555-blog/blob/main/static/images/A_5_years_brief_story_of_LLMs/T5_overview.png?raw=true" style= "display: block; margin-left: auto; margin-right: auto; width: 100%;"/>|
|:--:| 
| *T5 can perform multi-language tasks with prefix-conditioned task instruction <br> (Image from [])* |


## Architecture

In this paper, It has been proven that the *encoder-decoder* architecture outperforms *decoder-only* language models. **T5 follows *encoder-decoder* architecture** with the largest size of 11B parameters.


## Input Design

For self-supervised pre-trained stage, the input is designed in a following format to achieve **fill-in-the-blank-style denoising objectives.** The model was trained to recover only missing words in the input to reduce the computational cost.

| <img src="https://github.com/trapoom555/trapoom555-blog/blob/main/static/images/A_5_years_brief_story_of_LLMs/T5_input.png?raw=true" style= "display: block; margin-left: auto; margin-right: auto; width: 100%;"/>|
|:--:| 
| *Input and Target example in self-supervised pre-trained stage <br> (Image from [])* |

## Dataset

The authors created Colossal Clean Crawled Corpus (C4) dataset to train T5. It's a cleaned version of open-source web scraped dataset Common Crawl.

## Additional Insights

- Training on in-domain data can be beneficial but that pre-training on smaller datasets can lead to detrimental overfitting
- Multitask learning could be close to competitive with a pre-train-then-fine-tune approach but requires carefully choosing how often the model is trained on each task

# Scaling Law (2020.01)

The previous research gave us a signal that the performance of language modelling in Transformer models doesn't seem to stop if we continually increase the model size. But no one had studied its behavior systematically. This paper [] did emperical studies and discovered following insights.

- **Scaling Law** Model parameters $N$, size of dataset $D$ and amount of compute $C$ are major effects in performance while the model shape and hyperparameters effects performance very weakly.

- **Smooth Power Law** When scaling $N$ or $D$ or $C$ and fixing the other two, the performance doesn't bottlenecked by the other two.

- **Universality of Overfitting** If one or both of the $N$ or $D$ are fixed while scaling the others, the model will be overfitted.

- Generalization performance to other data distributions improves smoothly with model size $N$.

- Large models are more sample efficient than smaller models.

- If the compute budget $C$ is fixed, reaching the best performance can be done with a very large model.


# GPT-3 (2020.05)

When it was found that scaling the model size is the way to achieve generalization, this paper [] followed the trend and scaled GPT up to 175B parameters naming it as GPT3. By making the model large enough, the model can achieve a *few-shot learner* behavior. Meaning that it can perform all tasks by given *in-context examples* without any gradient updates or fine-tuning.

> **in-context examples** are examples or demonstrations, given in a prompt, of the task we wish the model to perform

Furthermore, It was demonstrated that the GPT3 with *few-shot learning* can surpass SOTA fine-tuned models in many benchmarks.

The *few-shot* settings can achieve higher performance than *zero-shot* which was demonstrated in GPT2 shown in the graph below. And it was also discovered that **Large models are more in-context examples efficient than smaller ones**.

| <img src="https://github.com/trapoom555/trapoom555-blog/blob/main/static/images/A_5_years_brief_story_of_LLMs/GPT3_acc_shot.png?raw=true" style= "display: block; margin-left: auto; margin-right: auto; width: 100%;"/>|
|:--:| 
| *Large models are more in-context examples efficient than small models <br> (Image from [])* |


| <img src="https://github.com/trapoom555/trapoom555-blog/blob/main/static/images/A_5_years_brief_story_of_LLMs/GPT3_acc_size.png?raw=true" style= "display: block; margin-left: auto; margin-right: auto; width: 90%;"/>|
|:--:| 
| *Accuracy vs. Model Size aggregated across benchmarks <br> (Image from [])* |

# FLAN (2021.09)

Fine-tuning has seen as the method to make the model more specialist but this study [] shows that the model can be more generalist using fine-tuning method called *Instruction Tuning.*

> **Instruction Tuning** is a fine-tuning method by using natural language instructions as prompts and answers as targets. Many NLP tasks are used in the instruction tuning.


| <img src="https://github.com/trapoom555/trapoom555-blog/blob/main/static/images/A_5_years_brief_story_of_LLMs/FLAN_comparison.png?raw=true" style= "display: block; margin-left: auto; margin-right: auto; width: 100%;"/>|
|:--:| 
| *Comparison among Instruction Tuning method, Fine-Tuning <br> and GPT3 few-shot prompting (Image from [])* |

FLAN uses LaMBDA-PT [], a pre-trained 137B *decoder-only* model, as a base model. Note that it's smaller than GPT3, but using this technique makes it more generaliable than the GPT3. For the diversity in the instruction tuning dataset, the authors construct 10 natural language templates for each NLP task. After tuning, the model surpasses *zero-shot* performance of GPT3 by 20 out of 25 unseen tasks.

| <img src="https://github.com/trapoom555/trapoom555-blog/blob/main/static/images/A_5_years_brief_story_of_LLMs/FLAN_performance.png?raw=true" style= "display: block; margin-left: auto; margin-right: auto; width: 100%;"/>|
|:--:| 
| *FLAN performance compared to GPT3 zero-shot and few-shot <br> (Image from [])* |

There're some interesting insights to note about FLAN.

- Adding more clusters in the instruction-tuning dataset make the model more generalist.

| <img src="https://github.com/trapoom555/trapoom555-blog/blob/main/static/images/A_5_years_brief_story_of_LLMs/FLAN_more_clusters.png?raw=true" style= "display: block; margin-left: auto; margin-right: auto; width: 70%;"/>|
|:--:| 
| *zero-shot performance adding new clusters in the instruction tuning dataset <br> (Image from [])* |

- Small model generalization can be hurt by using instruction-tuning method, potentially because all model capacity is used to learn the mixture of instruction tuning tasks.

| <img src="https://github.com/trapoom555/trapoom555-blog/blob/main/static/images/A_5_years_brief_story_of_LLMs/FLAN_small_model.png?raw=true" style= "display: block; margin-left: auto; margin-right: auto; width: 70%;"/>|
|:--:| 
| *Instruction Tuning only benefits for large enough models<br> (Image from [])* |

- Adding few-shot prompting (similar to GPT3) at inference time on an instruction-tuned model can improve the performance.

| <img src="https://github.com/trapoom555/trapoom555-blog/blob/main/static/images/A_5_years_brief_story_of_LLMs/FLAN_with_few_shot.png?raw=true" style= "display: block; margin-left: auto; margin-right: auto; width: 100%;"/>|
|:--:| 
| *instruction-tuned model with and without few-shot prompting <br> (Image from [])* |

# InstructGPT (2022.03)

At this point, the Language Models are more generalist, but their answers still don't align with *user inttents*. OpenAI solved this problem by proposing a simple but very useful method to train a model called InstructGPT [].

The method consist of 3 stages

| <img src="https://github.com/trapoom555/trapoom555-blog/blob/main/static/images/A_5_years_brief_story_of_LLMs/InstructGPT_steps.png?raw=true" style= "display: block; margin-left: auto; margin-right: auto; width: 100%;"/>|
|:--:| 
| *Three steps of training the InstructGPT <br> (Image from [])* |

1. **Supervised Fine-Tuning** Starting with the GPT3 [], Supervised fune-tuning (SFT) method is applied to train the model to mimic labeler's answers.

2. **Reward Model Training** Only SFT itself can push the model to be more align with user intents, but it will not be generalized for the unseen prompts. This method can be thought of a Behavioral Cloning in Imitation Learning. Therefore, the interactive learning using Reinforment Learning is adopted here. First, we are going to train a **Reward Model** which is initialized with a supervise fine-tuned 6B GPT3 model, replacing an unembedding layer with a projection layer, which maps the output to a corresponding scalar value. The data used to train the reward model is rankings of $K$ model's outputs. To optimize the Reward Model, we use the following loss function.
$$L(\theta) = -\frac{1}{K \choose 2} \mathbb{E}_{(x, y_w, y_l) \sim D} [ \log (\sigma (r _ \theta  (x, y_w) - r _ \theta (x, y_l) )) ]$$
where $r _ \theta (x, y)$ is the scalar output of the reward model for prompt $x$ and output $y$ with parameters $\theta$, $y_w$ is the *preferred completion* outof the pair of $y_w$ and $y_l$. The loss function will be minimized if the reward value of preferred answer is greater than another one as much as possible.
3. **Optimize a Policy against the Reward Model** The supervised fine-tuned model *interactively* updates its gradient by using **Proximal Policy Optimization (PPO)** using the trained Reward Model as a reward function.

The result on human preferences is impressive. With only 1.3B parameters, the InstructGPT outperforms a 175B GPT3 model.

| <img src="https://github.com/trapoom555/trapoom555-blog/blob/main/static/images/A_5_years_brief_story_of_LLMs/InstructGPT_results.png?raw=true" style= "display: block; margin-left: auto; margin-right: auto; width: 80%;"/>|
|:--:| 
| *Human preference results winrate against the 175B SFT model <br> (Image from [])* |

# PaLM (2022.04)
Google noticed that scaling the parameter size of a Large Language Model benefits the performance. **What if we further scale up the size ?** In this work, the model size was scaled to 540B which is called PaLM [] (Pathways Language Model).

PaLM is a *decoder-only* Transformer architecture with some modifications. It was trained on 780B tokens high-quality corpus with multilingual natural language and code.

| <img src="https://github.com/trapoom555/trapoom555-blog/blob/main/static/images/A_5_years_brief_story_of_LLMs/PaLM_data.png?raw=true" style= "display: block; margin-left: auto; margin-right: auto; width: 60%;"/>|
|:--:| 
| *High-quality data used to train PaLM (Image from [])* |


The followings are key takeaways from this paper.
1. They adopted Pathways [] for a large-scale training which enables training a single model across ten of thousands of accelerator chips in a highly efficient manner. (PaLM was trained on 6144 TPU v4 chips)

2. Up to 540B the performance continues to improve. There's no observation for the saturation point yet.

3. With this scale combined with chain-of-thought prompting [], the model outperform SOTA fine-tuned language model with simple few-shots.

4. It breaks the "power law". Scaling from 62B to 540B results in a drastic jump in accuracy compared to scaling from 8B to 62B.

5. It can do better job in multilingual tasks and can outperform prior SOTA translation tasks.

# OPT (2022.05)

OPT stands for Open Pre-trained Transformers. This paper [] aims to replicate GPT-3 and contribute model weights with size of 125M to 175B to research community. The model is a decoder-only Transformer. Training with 180B tokens of curated various open-source datasets, OPT achieves comparable performance to GPT-3 in zero-shot, one-shot and few-shot settings across 14 NLP tasks. The training process of OPT-175B consumes only 1/7th compared to GPT-3.

| <img src="https://github.com/trapoom555/trapoom555-blog/blob/main/static/images/A_5_years_brief_story_of_LLMs/OPT_performance.png?raw=true" style= "display: block; margin-left: auto; margin-right: auto; width: 60%;"/>|
|:--:| 
| *OPT and GPT-3 performance across 14 NLP tasks (Image from [])* |

# Emergent Abilities (2022.06)

> **Emergent Abilities** [] are abilities which are not present in smaller models but are present in larger ones

To be concise, the emergence of these abilities not only depends on model size, it also depends on amount of training computation, training data size and data quality.

The emergent abilities observed in this paper include
1. **Few-shot prompting** : giving few examples without fine-tuning makes model understand the task. By increasing the model size, at a certain point, the model learns the few-shot prompting ability.
| <img src="https://github.com/trapoom555/trapoom555-blog/blob/main/static/images/A_5_years_brief_story_of_LLMs/emergent_fewshot_prompting.png?raw=true" style= "display: block; margin-left: auto; margin-right: auto; width: 100%;"/>|
|:--:| 
| *Few-shot prompting performance over several benchmarks (Image from [])* |

2. **Augmented prompting strategies** (Other promting strategies)
    - <u>Multi-step reasoning</u> : chain-of-thought prompting [] (CoT)
    - <u>Instruction following</u> : fine-tuning the model with instruction prompting, the model shows ability to reponse unseen tasks appropriately.
    - <u>Program execution</u> : finetuning language models to predict intermediate outputs (“scratchpad”) enables them to successfully execute such multi-step computations.
    - <u>Model calibration</u> : a True/False technique, where models first propose answers and then evaluate the probability “P(True)” that their answers are correct.
| <img src="https://github.com/trapoom555/trapoom555-blog/blob/main/static/images/A_5_years_brief_story_of_LLMs/emergent_prompting_strategy.png?raw=true" style= "display: block; margin-left: auto; margin-right: auto; width: 100%;"/>|
|:--:| 
| *Specialized prompting or fine-tuning method can be emergent (Image from [])* |

# Scaling FLAN (2022.10)

Instruction Fine-tuning Language Model (FLAN) has shown a promissing result boosting the performance of LLM (2021.09). But at that time 10 tasks were used to fine-tune the model. Model's behavior on various intruction fine-tuning tasks has not yet to be observed. This paper [] aims to explore behaviors of LLMs scaling instruction fine-tuning tasks. Insights from this paper is listed below.

1. Scaling the number of tasks from 10 to 1.8K and model sizes increase the performance continually.

| <img src="https://github.com/trapoom555/trapoom555-blog/blob/main/static/images/A_5_years_brief_story_of_LLMs/Scale_FLAN_scale.png?raw=true" style= "display: block; margin-left: auto; margin-right: auto; width: 90%;"/>|
|:--:| 
| *Scaling effects on the number of tasks and model sizes (Image from [])* |

2. CoT critically improves reasoning ability. *Jointly* fine-tuning with CoT data improves the performance on evaluation dataset. It has been observed that training with just CoT data degrades model performance of the evaluation data which doesn't have CoT. 

| <img src="https://github.com/trapoom555/trapoom555-blog/blob/main/static/images/A_5_years_brief_story_of_LLMs/Scale_FLAN_CoT_Co_fine_tuning.png?raw=true" style= "display: block; margin-left: auto; margin-right: auto; width: 80%;"/>|
|:--:| 
| *Co-fine-tuning with CoT data improves performance (Image from [])* |

3. As discovered in the FLAN paper, doing instruction fine-tuning improves zero-shot ability. This finding is still true for larger model size and the number of tasks.

| <img src="https://github.com/trapoom555/trapoom555-blog/blob/main/static/images/A_5_years_brief_story_of_LLMs/Scale_FLAN_zero_shot.png?raw=true" style= "display: block; margin-left: auto; margin-right: auto; width: 50%;"/>|
|:--:| 
| *FLAN improves zero-shot ability (Image from [])* |

4. Several responsible AI benchmarks are improved using FLAN. It increases usability of LLMs.

5. All above insights work for both Decoder-Only and Encoder-Decoder models (PaLM and T5 were used in this paper for testing.) 

# References