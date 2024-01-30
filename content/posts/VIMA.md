---
author: "trapoom555"
title: "VIMA : Visuomotor Attention Agent"
date: "2024-01-30"
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

Many of manipulation tasks can be experssed as multimodal modal prompts. VIMA [[1]](#1) is a multimodal behavior-cloning manipulator agent. It processes multimodal prompt (text + image) and outputs discretized robot poses regressively. The output is conditioned by the multimodal prompt and interaction history.

## Architecture

| <img src="https://github.com/trapoom555/trapoom555-blog/blob/main/static/images/VIMA/architecture.png?raw=true" style= "display: block; margin-left: auto; margin-right: auto; width: 100%;"/>|
|:--:| 
| *VIMA Architecture (Image from [[1]](#1))* |

The prompt tokens are designed to repeatedly connect to the history tokens information path via interleaved **Cross-Attention (XATTN) layers**. The reasons behind this design are following
1. Strengthened connection to prompt
2. Deep flow of original prompt
3. Better computational efficiency


VIMA adopts object-centric representation which means it computes image tokens only for the interested objects. The tokens are computed from bounding box coordinates and cropped RGB patches.

### Bounding Boxes

The bounding boxes are obtained by applying a domain fine-tuned Mask RCNN [[2]](#2) to a tabletop scene.

### Tokenizer

- Text prompt : frozen T5 [[3]](#3) (111M parameters)
- Single object images : ViT
- Bounding boxes : MLP

### Action Decoder

Once the action token (the pale sky blue square in the picture above) is calculated, There'll be 6 independent MLPs which map the action token to a predicted robot pose.

The action space in VIMA is designed to be in $SE(2)$ which means the robot can only pick and place objects parallel to the table plane. In this paper the $SE(2)$ pose is defined by 6 parameters as follows

1. 2 variables for the position in x, y plane (position representation)
2. 4 vairables for quaternion (orientation representation)

## Training

### Data

VIMA-BENCH [[1]](#1) dataset is much larger than [PerAct](https://trapoom555.github.io/trapoom555-blog/posts/peract/) [[4]](#4). The dataset includes 50K trajectories per task, 650K successful trajectories in total. The dataset is designed to measure the zero-shot generalization of the model. If a model can perform well in a higher level tasks, it means that the model shows good results in generalization.

### Data Augmentation

Since Mask RCNN is inaccurate (it produces many dummy bounding boxes), the training data will be augmented to have false-positive object detection outputs to mimic the behavior of the Mask RCNN at inference time.

### Objective

For a demonstration with $T$ steps with gound-truth $a_t$, the objective is to maximize the following log-likelihood

$$\max_\theta \sum_{t=1}^T \pi_\theta (a_t \mid \mathcal P, \mathcal H)$$


### Resources

8 NVIDIA V100 GPUs. The largest experiment takes around 1 day of training.

## Performance

When evaluating on progressively harder settings, VIMA incurs much less performance drop over other baselines. And it can be concluded that VIMA has more zero-shot capability than other baselines.

| <img src="https://github.com/trapoom555/trapoom555-blog/blob/main/static/images/VIMA/performance.png?raw=true" style= "display: block; margin-left: auto; margin-right: auto; width: 60%;"/>|
|:--:| 
| *VIMA Performance vs. Other baselines (x-axis determines a level of task complexity) <br> (Image from [[1]](#1))* |

## Insights

- **ViT performs best** experimenting with different vision encoder. Note that the ViT (Oracle) uses ground-truth bounding box instead of using predicted ones from Mask RCNN.

| <img src="https://github.com/trapoom555/trapoom555-blog/blob/main/static/images/VIMA/performance_vision_encoder.png?raw=true" style= "display: block; margin-left: auto; margin-right: auto; width: 100%;"/>|
|:--:| 
| *VIMA Performance using different vision encoder (Image from [[1]](#1))* |

- **Prompt Conditioning** XATTN performs better than GPT-decoder in a small size model


| <img src="https://github.com/trapoom555/trapoom555-blog/blob/main/static/images/VIMA/performance_prompt_conditioning.png?raw=true" style= "display: block; margin-left: auto; margin-right: auto; width: 100%;"/>|
|:--:| 
| *VIMA Performance using XATTN and GPT-decoder (Image from [[1]](#1))* |

## Discussions and Limitations

- The Object detection model must be also used at the inference time which means the total memory size is even larger.
- The interested object must be known before doing the task in order to draw the bounding box.
- Output are still discretized robot poses that is similar to PerAct. This approach is not applicable to tasks which are sensitive to path.
- VIMA suffers from detection inaccuracy (Mask RCNN)
- It doesn't support $SE(3)$ tasks
- Using 6 variables for $SE(2)$ representation is redundant
- I found that the Vision Encoder plays an improtant role in success.

# Reference
<a id="1">[1]</a> 
Jiang, Yunfan, et al. "Vima: General robot manipulation with multimodal prompts." arXiv (2022).

<a id="2">[2]</a> 
He, K., Gkioxari, G., Doll´ar, P., and Girshick, R. Mask r-cnn. arXiv preprint arXiv: Arxiv-1703.06870, 2017.

<a id="3">[3]</a> 
Raffel, Colin, et al. "Exploring the limits of transfer learning with a unified text-to-text transformer." The Journal of Machine Learning Research 21.1 (2020): 5485-5551.

<a id="4">[4]</a> 
Shridhar, Mohit, Lucas Manuelli, and Dieter Fox. "Perceiver-actor: A multi-task transformer for robotic manipulation." Conference on Robot Learning. PMLR, 2023.
