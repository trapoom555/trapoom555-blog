---
author: "trapoom555"
title: "PerAct : Perceiver-Actor"
date: "2024-01-29"
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

# PerAct

Per-Act [[1]](#1) is a language-conditioned behavior-cloning agent for 6-DOF Manipulator. By using just a few expert demonstrations per task, PerAct outperforms previous methods like GATO [[2]](#2) which uses more demonstrations data.

> GATO is an image-to-action mapping which uses extreamly large expert's demonstrations set (15K episodes of block stacking, 94K episodes for Meta-world tasks).

The paper gives an argument that since PerAct applies 3D information as a feature instead of using only image data, it uses less demonstrations than an image-to-action mapping model.

## Framework

In PerAct settings, a task can be achieved by following keyframes sequentially. A keyframe consists of 6D pose of an end-effector, gripper action (open / close) and a variable which determine whether it is a collision avoidance action (yes / no). 

A model receives RGB-D observations and a natural language goal for the task and use these information to predict the keyframe. These process will be repeated until the goal is reached.

- Input : observed voxels (from RGB-D camera), natural language goal
- Output : keyframe


## Architecture

| <img src="https://github.com/trapoom555/trapoom555-blog/blob/main/static/images/PerAct/architecture.png?raw=true" style= "display: block; margin-left: auto; margin-right: auto; width: 100%;"/>|
|:--:| 
| *PerAct Architecture (Image from [[1]](#1))* |

### Language Encoder

[CLIP](https://trapoom555.github.io/trapoom555-blog/posts/clip/) [[3]](#3) language encoder is used as a language encoder in PerAct.

### Voxel Encoder and Decoder

Since the number of voxels is too large $(100^3)$, Perceiver Transformer [[4]](#4) is used to compress massive 3D voxel data to latent vectors of dimension $\mathbb R^{2048 \times 512}$ in order to reduce memory usage. After passing through the PerceiverIO Transformer, it will then upsampled into the original voxel grid $(100^3)$. The Perceiver Encoder and Decoder are connected in a U-net [[5]](#5) fashion.

### Get Predictions
The decoded voxels will be max-pooled and then decoded with linear layer to form their respective Q-function. The best action $\mathcal T$ can be obtained from each Q-function

$$\mathcal T_{trans} = \underset{(x,y,z)}{\operatorname{argmax\ }} Q_{trans} ((x,y,z) \mid v, I)$$

$$\mathcal T_{rot} = \underset{(\psi,\theta, \phi)}{\operatorname{argmax\ }} Q_{rot} ((\psi,\theta, \phi) \mid v, I)$$

$$\mathcal T_{open} = \underset{\omega}{\operatorname{argmax\ }} Q_{open} (\omega \mid v, I)$$

$$\mathcal T_{collide} = \underset{\kappa}{\operatorname{argmax\ }} Q_{collide} (\kappa \mid v, I)$$

> Note : I skipped many details here. The key takeaway from this architecture is that the ground-truth can be discretized and we can take the problem as a simple classification problem.

## Training

### Data
The data comes from the expert's demonstrations which consists of voxel observations, language goals and keyframe
$\mathcal D=\\{ (v_1, l_1, k_1), (v_2, l_2, k_2), ...\\}$ 

In this paper, there're two experiments. The first one is the experiment performed in a simulation and the second one is performed in a real environment.

- in-simulation experiment : 100 demonstrations for training set, 25 for validation set, 25 for test set
- in-real-environment experiment : 57 demonstrations in total

The keyframe ground-truths are one-hot encoded and discretized as follows

1. Translation : discretized voxel grid $\mathbb R^{w\times h\times d}$
2. Rotation : discretized to $R$ bins $\mathbb R^{(360/R) \times 3}$
3. Gripper Action : $\mathbb R^2$
4. Collide Action : $\mathbb R^2$

### Data Augmentation

The data augmentation can be done with
$\pm 0.125 $ m in a translation perturbations, $\pm 45$ degree in a yaw rotation direction perturbations.

### Loss Function

Cross-entropy (take the problem as a classification problem)

### Resources

- in-simulation experiment : 8 V100 GPUs for 16 days (600K iteration)
- in-real-environment experiment : 8 P100 GPUs for 2 days
- Inference : a single TitanX GPU

## Performance

| <img src="https://github.com/trapoom555/trapoom555-blog/blob/main/static/images/PerAct/performance_sim.png?raw=true" style= "display: block; margin-left: auto; margin-right: auto; width: 100%;"/>|
|:--:| 
| *PerAct Performance in simulation (Image from [[1]](#1))* |

| <img src="https://github.com/trapoom555/trapoom555-blog/blob/main/static/images/PerAct/performance_real.png?raw=true" style= "display: block; margin-left: auto; margin-right: auto; width: 60%;"/>|
|:--:| 
| *PerAct Performance on 7 real-world tasks (Image from [[1]](#1))* |

## Discussions and Limitations

- CLIP vision encoder can be used in future work to align text and image for more generalization to unseen categories and instances.
- They don't freeze the CLIP weight ?
- Dexterious controls remains a challenge. It's unclear that PerAct framework will perform well in that kind of task (e.g. Folding cloth)
- Since PerAct is an action sample-based method, it cannot address the problem that requires continuous sensitive path like pouring water into a cup and also it doesn't support real-time close-loop control.
- It doesn't generalize well. According to the exeriment, changing a drawer color and texture can make robot confuse.
- Data Augmentation doesn't consider the kinematics feasibility. The actual robot may cannot reach some poses.

# Reference
<a id="1">[1]</a> 
Shridhar, Mohit, Lucas Manuelli, and Dieter Fox. "Perceiver-actor: A multi-task transformer for robotic manipulation." Conference on Robot Learning. PMLR, 2023.

<a id="2">[2]</a> 
Reed, Scott, et al. "A generalist agent." arXiv preprint arXiv:2205.06175 (2022).

<a id="3">[3]</a> 
Radford, Alec, et al. "Learning transferable visual models from natural language supervision." International conference on machine learning. PMLR, 2021.

<a id="4">[4]</a> 
A. Jaegle, S. Borgeaud, J.-B. Alayrac, C. Doersch, C. Ionescu, D. Ding, S. Koppula, D. Zoran,
A. Brock, E. Shelhamer, et al. Perceiver io: A general architecture for structured inputs &
outputs. arXiv preprint arXiv:2107.14795, 2021.

<a id="5">[5]</a> 
Ronneberger, Olaf, Philipp Fischer, and Thomas Brox. "U-net: Convolutional networks for biomedical image segmentation." Medical Image Computing and Computer-Assisted Intervention–MICCAI 2015: 18th International Conference, Munich, Germany, October 5-9, 2015, Proceedings, Part III 18. Springer International Publishing, 2015.