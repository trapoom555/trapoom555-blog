---
author: "trapoom555"
title: "[Draft] PerAct : PERCEIVER-ACTOR"
date: "2024-01-28"
math: mathjax
draft: true
tags: [
    "multi-modal",
    "robot-learning",
    "manipulator"
]
---

- Per-Act is a language-conditioned behavior-cloning agent for 6-DOF Manipulation
- With the right problem formulation, the manipulation task can benefit from Transformers
- Input : Voxel & Language
- Output : 3D Translation, 3D Rotation, Gripper Action
- These outputs will be repeated until the goal is reached
- Use just a few demonstrations per task
- In total, 7 real-world task uses only 53 demonstrations
- PerAct uses BC instead of RL which can be easily trained the model with language conditioned
- PerAct uses 3D voxels as a feature, it uses less demonstrations than image-to-action mapping
- GATO is an image-to-action mapping which uses extreamly large expert's demonstrations set (15K episodes of block stacking, 94K episodes for Meta-world tasks)
- Architecture
    - CLIP language encoder is used as a language encoding in PerAct
    - Perceiver Transformer is used to compress massive 3D voxel data to latent vectors $\mathbb R^{2048 \times 512}$ in order to reduce memory usage
    - It will then upsampled into the original voxel grid $(100^3)$
    - The voxels will be max-pooled and then decoded with linear layer to form their respective Q-function
    - The best action $\mathcal T$ can be obtained from each Q-function

$$\mathcal T_{trans} = \underset{(x,y,z)}{\operatorname{argmax\ }} Q_{trans} ((x,y,z) \mid v, I)$$

$$\mathcal T_{rot} = \underset{(\psi,\theta, \phi)}{\operatorname{argmax\ }} Q_{rot} ((\psi,\theta, \phi) \mid v, I)$$

$$\mathcal T_{open} = \underset{\omega}{\operatorname{argmax\ }} Q_{open} (\omega \mid v, I)$$

$$\mathcal T_{collide} = \underset{\kappa}{\operatorname{argmax\ }} Q_{collide} (\kappa \mid v, I)$$

- Training 
    - Data : the data comes from the expert's demonstrations which consists of voxel observations, language goals and keyframe
    $\mathcal D=\{ (v_1, l_1, k_1), (v_2, l_2, k_2), ...\}$ 
    Simulation : 100 demonstrations for training, 25 for val, 25 for test
    Real : 57 demonstrations
    - The agent tries to predict the keyframe $k$ given voxels $v$ and language goals $l$ 
    - The keyframe ground-truth will be one-hot encoded and discretized as follows
        - Translation : voxel grid $\mathbb R^{w\times h\times d}$
        - Rotation : discretize to $R$ bins $\mathbb R^{(360/R) \times 3}$
        - isOpen : $\mathbb R^2$
        - isCollide : $\mathbb R^2$
    
    - Data Augmentaiton : +- 1.25 m translation, +- 45 degree yaw rotation

    - Loss : Cross-entropy (take it as a classification problem)

    - Resources : 8 V100 GPUs for 16 days (600K iteration) (for simulation data)
    8 P100 GPUs for 2 days (for real robot)

- Inference : a single TitanX GPU

- Discussion
    - CLIP vision encoder can be used in future work to align text and image for more generalization to unseen categories and instances
    - They don't freeze the weight CLIP?
    - Dexterious controls remains a challenge. It's unclear that PerAct framework will perform well (e.g. Folding cloth)
    - Since PerAct is an action sample-based method, it cannot address the problem that requires continuous sensitive path like pouring water into a cup and also it doesn't support real-time close-loop control
    - It doesn't generalize well. According to the exeriment, changing the drawer color and texture make robot confuse
    - Data Augmentation doesn't consider the Kinematics Feasibility
