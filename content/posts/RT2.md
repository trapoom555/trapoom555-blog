---
author: "trapoom555"
title: "RT-2 : Robot Transformer 2"
date: "2024-02-03"
math: mathjax
tags: [
    "multimodal",
    "robot-learning",
    "manipulator",
    "paper-review",
    "computer-vision",
    "NLP",
    "transfer-learning"
]
---

## Framework

By taking robot actions as text tokens, RT-2 [[1]](#1) normalizes the data format. It does not only rely on robotics data but also takes advantages of large-scale multimodal internet data. It is also shown that incoperating chain of thought [[2]](#2) reasoning, RT-2 can perform multi-stage reasoning.

## Architecture

RT-2 uses vision-language model (VLM) as a backbone. PALI-X [[3]](#3) 5B & 55B models and [PaLM-E](https://trapoom555.github.io/trapoom555-blog/posts/palme/) [[4]](#4) 12B model have been selected as the backbone for the experiment. These two variants are named as following.

- **RT-2-PaLI-X** (built from PaLI-X 5B & 55B)
- **RT-2-PaLM-E** (built from PaLM-E 12B)

| <img src="https://github.com/trapoom555/trapoom555-blog/blob/main/static/images/RT2/architecture.png?raw=true" style= "display: block; margin-left: auto; margin-right: auto; width: 100%;"/>|
|:--:| 
| *RT-2 Architecture (Image from [[1]](#1))* |

### Output Action Token

The output actions are in the following format

$$\text{"terminate} \quad \Delta \text{pos}_x \quad \Delta \text{pos}_y \quad \Delta \text{pos}_z \quad \Delta \text{rot}_x \quad \Delta \text{rot}_y \quad \Delta \text{rot}_z \quad \text{gripper_extension"}$$

Similar to [RT-1](https://trapoom555.github.io/trapoom555-blog/posts/rt1/) [[5]](#5), robot actions are discretized into 256 bins. Each bin acts as a robot action token and will be assigned to the existing VLM's tokens. The methods of assignation are different for PaLI-X and PaLM-E

- PaLI-X : According to its tokenization, integers up to 1000 each have a unique token, so the action bins are assigned to the token representing the corresponding integer.
- PaLM-E : Overwrite least frequent used tokens.


## Training

In the experiment, RT-2 uses pre-trained base models and co-fine-tuning them with VQA and robot action data.

### Data

1. Internet-scale VQA
2. Robot Action Data (the same data that RT-1 has been trained on)

### Co-Fine-Tuning

An important detail of training recipe is that two sources of data (VQA & robot) are used during fine-tuning. As the training prosess goes, the proportion of the robot data is increased.

### Output Constraint

Since the robot action is needed to be in the correct format, RT-2 guarantees the correctness by **constraining its output vocabulary** via only sampling valid action tokens when model is prompted with a robot-action task. But in the VQA task, all vocabularies are allowed.


### Loss

Categorical cross-entropy loss (Next token prediction objective)

### Resources

*Not mentioned*

## Inference

### Resources

It is infeasible running 55B model on a single computer therefore, the model is deployed on **multi-TPU** cloud service.

### Control-loop frequency

- Big models with 55B parameters : $1\text{-}3\text{ Hz}$

- Smaller models with 5B parameters : $5 \text{ Hz}$

## Performance

The evaluations are mainly focused on an aspect of real-world generalization. RT-2 performs as good as RT-1 in seen tasks but has much more generalization ability than the RT-1.

| <img src="https://github.com/trapoom555/trapoom555-blog/blob/main/static/images/RT2/performance_generalization.png?raw=true" style= "display: block; margin-left: auto; margin-right: auto; width: 100%;"/>|
|:--:| 
| *Evaluation on Generalization (Image from [[1]](#1))* |

RT-2 can aquire knowledge form VLM and shows emergent capabilities (new ability beyond demonstraions) over RT-1 which is trained on only robot data from scratch.

| <img src="https://github.com/trapoom555/trapoom555-blog/blob/main/static/images/RT2/performance_emergent_skill.png?raw=true" style= "display: block; margin-left: auto; margin-right: auto; width: 80%;"/>|
|:--:| 
| *Evaluation on Emergent Capabilities (Image from [[1]](#1))* |

## Insights

- Co-fine-tuning can boost the performance.
- Model parameters largely affects the performance. This insight corresponds to the PaLM-E paper pointing out that  the catastrophic forgetting affects more in smaller models.

| <img src="https://github.com/trapoom555/trapoom555-blog/blob/main/static/images/RT2/performance_parameters.png?raw=true" style= "display: block; margin-left: auto; margin-right: auto; width: 80%;"/>|
|:--:| 
| *Impact of parameter counts and training strategy on generalization (Image from [[1]](#1))* |

- RT-2 reasoning ability can be benefit from chain-of-thought technique. By adding the additional data telling the steps to the model and further fine-tune the model with a few hundreds steps, the model performs better in reasoning.

| <img src="https://github.com/trapoom555/trapoom555-blog/blob/main/static/images/RT2/cot.png?raw=true" style= "display: block; margin-left: auto; margin-right: auto; width: 100%;"/>|
|:--:| 
| *COT can improve reasoning skills (Image from [[1]](#1))* |


## Discussions and Limitations

- RT-2 has shown a potential of transfering VLM knowledge to a robot.
- The model is too large to fit in any edge devices.
- Physical skills are limited to robot data. It could be nice if we can learn through new data collection paradigms e.g. from humans videos.
- There're not many open-source VLMs.
- Castastrophic forgetting destroy generalization for smaller models.

## References

<a id="1">[1]</a> 
Brohan, Anthony, et al. "Rt-2: Vision-language-action models transfer web knowledge to robotic control." arXiv preprint arXiv:2307.15818 (2023).

<a id="2">[2]</a> 
Wei, Jason, et al. "Chain-of-thought prompting elicits reasoning in large language models." Advances in Neural Information Processing Systems 35 (2022): 24824-24837.

<a id="3">[3]</a> 
Chen, Xi, et al. "PaLI-X: On Scaling up a Multilingual Vision and Language Model." arXiv preprint arXiv:2305.18565 (2023).

<a id="4">[4]</a> 
Driess, Danny, et al. "Palm-e: An embodied multimodal language model." arXiv preprint arXiv:2303.03378 (2023).

<a id="5">[5]</a> 
Brohan, Anthony, et al. “Rt-1: Robotics transformer for real-world control at scale.” arXiv preprint arXiv:2212.06817 (2022).
