---
author: "trapoom555"
title: "LAVA : Talking to Robots in Real Time"
date: "2024-02-02"
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

LAVA (Language Attends to Vision to Act) [[1]](#1) takes images and a natural language goal as its input and then output a robot action with $5 \text{ Hz}$ rate. It learns by behavioral cloning method and achieves $93.5\\%$ success rate.

| <img src="https://github.com/trapoom555/trapoom555-blog/blob/main/static/images/LAVA/framework.png?raw=true" style= "display: block; margin-left: auto; margin-right: auto; width: 100%;"/>|
|:--| 
| *Stage 1 : data collection by experts <br> Stage 2 : Train the LAVA model by behavioral cloning method <br> Stage 3 : Inference at $5 \text{ HZ}$ rate (Image from [[1]](#1))* |

## Architecture

| <img src="https://github.com/trapoom555/trapoom555-blog/blob/main/static/images/LAVA/architecture.png?raw=true" style= "display: block; margin-left: auto; margin-right: auto; width: 70%;"/>|
|:--:| 
| *LAVA architecture (Image from [[1]](#1))* |

- **Image Encoder** adopts ImageNet-pretrained ResNet [[2]](#2), [[3]](#3) which produces multi-scale visual features. These features introduce scale-invariant property to the system.
- **Language Encoder** applies [CLIP](https://trapoom555.github.io/trapoom555-blog/posts/clip/)'s [[4]](#4) language encoder.
- **Vision Language Fusion** is a [Transformer](https://trapoom555.github.io/trapoom555-blog/posts/transformer/) [[5]](#5) which perform cross-attention between language query and flattened multi-scale visual tokens acting as keys and values.
- **Rollup over time** uses a temperal Transformer [[6]](#6) which recieves 4 history of visual language tokens and produce an embedding for a policy.
- **Policy** is a deep residual connection MLP that outputs an action.


## Training

### Data

The paper also contributes a *Language-Table* dataset which consists of 181K simulation demos and 413K real demos. In total, the dataset comprises nearly 600K samples and nearly 200K samples are unique tasks.

### Loss

Mean square error $\sum_{(s, a, l) \sim \mathcal D_{\text{training}}} \Vert a - \pi_\theta (s, l) \Vert_2^2$ is used as a loss function where $a$ is a 2D cartesian setpoint of expert's demonstrations.

### Resources

TPUv3 8x8 pod (64 TPUv3 chips), 18 hours

## Performance

LAVA achieves $93.5\\%$ success rate on average in real-world tasks.

| <img src="https://github.com/trapoom555/trapoom555-blog/blob/main/static/images/LAVA/performance_real_world.png?raw=true" style= "display: block; margin-left: auto; margin-right: auto; width: 60%;"/>|
|:--:| 
| *LAVA performance in real-world tasks (Image from [[1]](#1))* |

## Insights

- LAVA architecture is compared to the ResNET-18 FiLM model [[7]](#7) showing that the SPL value of ResNET-18 FiLM is much lower than the LAVA while the success rate is almost equal. This means that the ResNET-18 FiLM doesn't take optimal paths.

| <img src="https://github.com/trapoom555/trapoom555-blog/blob/main/static/images/LAVA/performance_sim.png?raw=true" style= "display: block; margin-left: auto; margin-right: auto; width: 100%;"/>|
|:--:| 
| *LAVA, RestNet-18 FiLM success rate and SPL in a simulation test  (Image from [[1]](#1))* |

> **Success weighted by Path Length (SPL)** [[8]](#8) is a measurement of agent's navigation performance which is defined by
> $$\frac{1}{N} \sum_{i=1}^N S_i \frac{l_i}{\max(l_i, p_i)}$$
> where $l_i$ be the shortest path distance between starting position and goal in the episode $i$, $p_i$ is the actual distance between starting position and goal in the episode $i$ and $S_i$ is a binary indicator of success in the episode $i$.

## Discussions and Limitations

- The output actions are not discretized keypoints as they are in [PerAct](https://trapoom555.github.io/trapoom555-blog/posts/peract/) [[9]](#9), [VIMA](https://trapoom555.github.io/trapoom555-blog/posts/vima/) [[10]](#10) and [RT-1](https://trapoom555.github.io/trapoom555-blog/posts/rt1/) [[11]](#11). This paper has shown the possibility of using continuous value as prediction.

- According to the paper, the model could be developed further to be ablr to interact with humans with non-verbal settings and the future work can be deploying this model to a real assistant robot.

## References

<a id="1">[1]</a> 
C. Lynch et al., "Interactive Language: Talking to Robots in Real Time," in IEEE Robotics and Automation Letters, doi: 10.1109/LRA.2023.3295255.

<a id="2">[2]</a> 
J. Deng, W. Dong, R. Socher, L.-J. Li, K. Li, and L. Fei-Fei, “Imagenet: A large-scale hierarchical image database,” in 2009 IEEE conference on computer vision and pattern recognition. Ieee, 2009, pp. 248–255.

<a id="3">[3]</a>
K. He, X. Zhang, S. Ren, and J. Sun, “Deep residual learning for image recognition,” in Proceedings of the IEEE conference on computer vision and pattern recognition, 2016, pp. 770–778.

<a id="4">[4]</a> 
Radford, Alec, et al. “Learning transferable visual models from natural language supervision.” International conference on machine learning. PMLR, 2021.

<a id="5">[5]</a> 
Vaswani, Ashish, et al. “Attention is all you need.” Advances in neural information processing systems 30 (2017).

<a id="6">[6]</a> 
R. Xiong, Y. Yang, D. He, K. Zheng, S. Zheng, C. Xing, H. Zhang, Y. Lan, L.Wang, and T. Liu, “On layer normalization in the transformer architecture,” in International Conference on Machine Learning. PMLR, 2020, pp.10 524–10 533.

<a id="7">[7]</a> 
E. Jang, A. Irpan, M. Khansari, D. Kappler, F. Ebert, C. Lynch, S. Levine, and C. Finn, “Bc-z: Zero-shot task generalization with robotic imitation learning,” in Conference on Robot Learning. PMLR, 2022, pp. 991–1002.

<a id="8">[8]</a> 
P. Anderson, A. Chang, D. S. Chaplot, A. Dosovitskiy, S. Gupta, V. Koltun, J. Kosecka, J. Malik, R. Mottaghi, M. Savva et al., “On evaluation of embodied navigation agents,” arXiv preprint arXiv:1807.06757, 2018.

<a id="9">[9]</a>
Shridhar, Mohit, Lucas Manuelli, and Dieter Fox. “Perceiver-actor: A multi-task transformer for robotic manipulation.”Conference on Robot Learning. PMLR, 2023.

<a id="10">[10]</a>
Jiang, Yunfan, et al. “Vima: General robot manipulation with multimodal prompts.” arXiv (2022).

<a id="11">[11]</a>
Brohan, Anthony, et al. “Rt-1: Robotics transformer for real-world control at scale.” arXiv preprint arXiv:2212.06817 (2022).