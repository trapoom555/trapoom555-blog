---
author: "trapoom555"
title: "PALM-E : PALM-Embodied"
date: "2024-02-01"
math: mathjax
tags: [
    "multimodal",
    "robot-learning",
    "paper-review",
    "computer-vision",
    "NLP",
    "transfer-learning"
]
---

## Framework

PALM-E (562 B) [[1]](#1) combines PALM (540 B) [[2]](#2) language model and [ViT](https://trapoom555.github.io/trapoom555-blog/posts/vit/) (22 B) [[3]](#3) vision model together. It is trained by mixing embodied data into the training of multimodal large language model. The model can recieves 3 types of interleaved data as a prompt including *texts, images and states* and output **high level planning** as *texts* regressively.

PALM-E is a generalist vision language model. It not only can do the robot manipulation task, but it also can do other tasks, for example, Vision Q&A, Captioning, Language only task and more.

## Architecture

The main idea is quite similar to the [Frozen](https://trapoom555.github.io/trapoom555-blog/posts/frozen/) [[4]](#4) model which tries to map multimodal data into sequence of tokens with the same dimension before feeding them into a Large Language Model (LLM). 

| <img src="https://github.com/trapoom555/trapoom555-blog/blob/main/static/images/PALME/architecture.png?raw=true" style= "display: block; margin-left: auto; margin-right: auto; width: 100%;"/>|
|:--:| 
| *PALM-E architecture (Image from [[1]](#1))* |

## Training

### Data

Alongside the multimodal data, the following embodied data are used for training

1. TAMP (Task and Motion Planning)
2. Language-Table [[5]](#5) 
3. SayCan [[6]](#6) 

### Loss

Categorical cross-entropy

### Resources

*Not mentioned* (but It's definitely a hardcore one 🥶)

## Performance

Training on multiple tasks, PALM-E exhibits variety of capabilities including
1. Zero-shot multimodal chain-of-thought reasoning (before this paper published, COT is originally a language-only concept)
2. Few-shot prompting
3. OCR-free math reasoning
4. Multi-image reasoning (despite training with a single image)

| <img src="https://github.com/trapoom555/trapoom555-blog/blob/main/static/images/PALME/multitask.png?raw=true" style= "display: block; margin-left: auto; margin-right: auto; width: 100%;"/>|
|:--:| 
| *PALM-E can perform multitasks (Image from [[1]](#1))* |

The performance in high-level planning is measured in TAMP environment. The paper did the experimentation on variants of PALM-E by changing an input encoder or input format and also compare it to the other baselines including SayCan and PALI [[7]](#7).

| <img src="https://github.com/trapoom555/trapoom555-blog/blob/main/static/images/PALME/TAMP_performance.png?raw=true" style= "display: block; margin-left: auto; margin-right: auto; width: 100%;"/>|
|:--:| 
| *PALM-E performance on the TAMP environment (Image from [[1]](#1))* |

- **State** is a description of each object given by the TAMP environment. It consists of pose, size, color etc. The states are vectorized and fed into MLP to map into the language embedding space.

- **ViT + TL with Object Centric** the image is used as a feature instead of states. By doing object centric, the image will be masked by object producing multiple images corresponding to each object which will be passed through the ViT. After getting a tokens from the ViT, the TokenLearner [[8]](#8) is used to compress the token to the specific size. Note that this method requires mask *ground-truth*.

- **ViT-4B** doesn't require any mask ground-truth. ViT takes an image to produce a token. The token will be transformed into the language embedding space by using a simple affine transformation. This method isn't an object-centric approach. It can be hard for the model to learn this representation.

- **OSRT** Object Scene Representation Transformer (OSRT) [[9]](#9) decomposes scenes into individual objects without supervision. Meaning that, it doesn't require ground-truth and also it's an object-centric approach. The tokens produced by OSRT will be then passed through MLP producing final tokens for the LLM. The experimentation shows that OSRT is the best way to represent scene despite using a few data.

## Working with real robots

PALM-E acts as a high-level planner which output a high-level natural language planning, or in other words, it produces sub-goals from user's goal at $1\text{ Hz}$ to the policy from "Interactive Language: Talking to Robots in Real Time" [[5]](#5) which output the low-level robot action at $5\text{ Hz}$

Before deploying policy to the real robot, it is fine-tuned on 100 long horizon tasks with one example on each task. The policy can generalize zero-shot to new objects !

| <img src="https://github.com/trapoom555/trapoom555-blog/blob/main/static/images/PALME/performance_robot.png?raw=true" style= "display: block; margin-left: auto; margin-right: auto; width: 100%;"/>|
|:--:| 
| *PALM-E on a real robot (Image from [[1]](#1))* |

## Insights

- Multi-task training improves performance compared to the individual task training

| <img src="https://github.com/trapoom555/trapoom555-blog/blob/main/static/images/PALME/mix_data.png?raw=true" style= "display: block; margin-left: auto; margin-right: auto; width: 60%;"/>|
|:--:| 
| *PALM-E performance gained when training with multiple tasks (Image from [[1]](#1))* |

- Scaling the language model size is proven to prevent the *catastrophic forgetting* when fine-tune with multimodal data. **This is the reason why PALM-E is so big.**

| <img src="https://github.com/trapoom555/trapoom555-blog/blob/main/static/images/PALME/catastrophic_forget_model_size.png?raw=true" style= "display: block; margin-left: auto; margin-right: auto; width: 60%;"/>|
|:--:| 
| *The performance on general language tasks drop less when increase the model size (Image from [[1]](#1))* |

- Finetune LLM with the data from all sources (full mixture) perform best.

| <img src="https://github.com/trapoom555/trapoom555-blog/blob/main/static/images/PALME/LLM_finetune_or_frozen.png?raw=true" style= "display: block; margin-left: auto; margin-right: auto; width: 60%;"/>|
|:--:| 
| *Performance Frozen vs Fine-tune LLM and single vs full mixture data (Image from [[1]](#1))* |

- Input representation is very important to the performance !

- The PALI baseline can't perform the Embodied-AI tasks.

## Discussions and Limitations

- The model size is large (562 B). Inferencing and maintaining it can be costly.
- High level planning as texts instead of action can be applicable to different domains of robots e.g. manipulator, mobile manipulator.
- For some specific tasks like robot manipulation, the other capabilities of PALM-E aren't required. The model can be reduced to be smaller for the specific application.
- Natural Language output for robot as a high level planning may vary a lot. It requires translation to the robot's action. In this case another model was used. It makes the whole system even larger.

## References

<a id="1">[1]</a> 
Driess, Danny, et al. "Palm-e: An embodied multimodal language model." arXiv preprint arXiv:2303.03378 (2023).

<a id="2">[2]</a> 
Chowdhery, Aakanksha, et al. "Palm: Scaling language modeling with pathways." Journal of Machine Learning Research 24.240 (2023): 1-113.

<a id="3">[3]</a>
Dosovitskiy, Alexey, et al. "An image is worth 16x16 words: Transformers for image recognition at scale." arXiv preprint arXiv:2010.11929 (2020).

<a id="4">[4]</a> 
Tsimpoukelli, Maria, et al. "Multimodal few-shot learning with frozen language models." Advances in Neural Information Processing Systems 34 (2021): 200-212.

<a id="5">[5]</a> 
Lynch, C., Wahid, A., Tompson, J., Ding, T., Betker, J., Baruch, R., Armstrong, T., and Florence, P. Interactive language: Talking to robots in real time. arXiv preprint arXiv:2210.06407, 2022.

<a id="6">[6]</a> 
Ahn, M., Brohan, A., Brown, N., Chebotar, Y., Cortes, O., David, B., Finn, C., Gopalakrishnan, K., Hausman, K., Herzog, A., et al. Do as i can, not as i say: Grounding language in robotic affordances. arXiv preprint arXiv:2204.01691, 2022.

<a id="7">[7]</a> 
Chen, Xi, et al. "Pali: A jointly-scaled multilingual language-image model." arXiv preprint arXiv:2209.06794 (2022).

<a id="8">[8]</a> 
Michael Ryoo, AJ Piergiovanni, Anurag Arnab, Mostafa Dehghani, and Anelia Angelova. Tokenlearner:
Adaptive space-time tokenization for videos. Advances in Neural Information Processing
Systems, 34:12786–12797, 2021.

<a id="9">[9]</a> 
Sajjadi, Mehdi SM, et al. "Object scene representation transformer." Advances in Neural Information Processing Systems 35 (2022): 9512-9524.