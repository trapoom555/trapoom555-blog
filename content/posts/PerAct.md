---
author: "trapoom555"
title: "PerAct : Perceiver-Actor"
date: "2024-01-28"
math: mathjax
draft: true
tags: [
    "multi-modal",
    "robot-learning",
    "manipulator",
    "paper-review"
]
---

# PerAct

Per-Act is a language-conditioned behavior-cloning agent for 6-DOF Manipulator. By using just a few expert demonstrations per task, PerAct outperforms previous methods like GATO [] which uses more demonstrations data.

> GATO is an image-to-action mapping which uses extreamly large expert's demonstrations set (15K episodes of block stacking, 94K episodes for Meta-world tasks).

The paper gives an argument that since PerAct applies 3D information as a feature instead of using only image data, it uses less demonstrations than an image-to-action mapping model.

## Framework

In PerAct settings, a task can be achieved by following keyframes sequentially. A keyframe consists of 6D pose of an end-effector, gripper action (open / close), is it a collision avoidance action (yes / no). 

A model receives RGB-D observations and a natural language goal for the task and use these information to predict the keyframe. These process will be repeated until the goal is reached.

- Input : observed voxels (from RGB-D camera), natural language goal
- Output : keyframe


## Architecture

| <img src="https://github.com/trapoom555/trapoom555-blog/blob/main/static/images/PerAct/architecture.png?raw=true" style= "display: block; margin-left: auto; margin-right: auto; width: 100%;"/>|
|:--:| 
| *PerAct Architecture (Image from [[1]](#1))* |

### Language Encoder

CLIP [] language encoder is used as a language encoder in PerAct.

### Voxel Encoder and Decoder

Since the number of voxels is too large $(100^3)$, Perceiver Transformer [] is used to compress massive 3D voxel data to latent vectors of dimension $\mathbb R^{2048 \times 512}$ in order to reduce memory usage. After passing through the PercieverIO Transformer, it will then upsampled into the original voxel grid $(100^3)$. The Perceiver Encoder and Decoder are connected in a U-net [] fashion.

### Get Predictions
The decoded voxels will be max-pooled and then decoded with linear layer to form their respective Q-function. The best action $\mathcal T$ can be obtained from each Q-function

$$\mathcal T_{trans} = \underset{(x,y,z)}{\operatorname{argmax\ }} Q_{trans} ((x,y,z) \mid v, I)$$

$$\mathcal T_{rot} = \underset{(\psi,\theta, \phi)}{\operatorname{argmax\ }} Q_{rot} ((\psi,\theta, \phi) \mid v, I)$$

$$\mathcal T_{open} = \underset{\omega}{\operatorname{argmax\ }} Q_{open} (\omega \mid v, I)$$

$$\mathcal T_{collide} = \underset{\kappa}{\operatorname{argmax\ }} Q_{collide} (\kappa \mid v, I)$$

> Note : I skipped many details here. The key takeaway from this architecture is that the ground-truth can be discretized and we can take the problem as a simple classification problem.

## Training

### Data
the data comes from the expert's demonstrations which consists of voxel observations, language goals and keyframe
$\mathcal D=\\{ (v_1, l_1, k_1), (v_2, l_2, k_2), ...\\}$ 

In this paper, there're two experiments. The first one is in a simulation and the second one is in a real environment.

- Simulation experiment : 100 demonstrations for training, 25 for val, 25 for test
- Real environment experiment : 57 demonstrations

The keyframe ground-truth will be one-hot encoded and discretized as follows

1. Translation : voxel grid $\mathbb R^{w\times h\times d}$
2. Rotation : discretized to $R$ bins $\mathbb R^{(360/R) \times 3}$
3. Gripper Action : $\mathbb R^2$
4. Collide Action : $\mathbb R^2$

### Data Augmentation

The data augmentation can be done my moving the keypoints by
$\pm 0.125 $ m in a translation, $\pm 45$ degree in a yaw rotation direction.

### Loss Function

Cross-entropy (take the problem as a classification problem)

### Resources

- Simulation experiment : 8 V100 GPUs for 16 days (600K iteration)
- Real environment experiment : 8 P100 GPUs for 2 days
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
- It doesn't generalize well. According to the exeriment, changing the drawer color and texture can make robot confuse.
- Data Augmentation doesn't consider the kinematics feasibility. The actual robot may cannot reach some pose.

# Reference

