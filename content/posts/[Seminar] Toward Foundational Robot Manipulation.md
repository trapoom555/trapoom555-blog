---
author: "trapoom555"
title: "[Seminar] Toward Foundational Robot Manipulation Skills"
date: "2024-01-28"
tags: [
    "seminar",
    "robot-learning"
]
---

## Details
Organizer: MIT Robotics <br>
Speaker: Dieter Fox [Washington U. / NVIDIA] <br>
Date: 22-05-2023 <br>
Link: [Youtube](https://www.youtube.com/watch?v=1EFZ--nbKog&t=2556s)

## Notes
- In his point of view, Behavior Cloning in robotics is similar to a LLM's self-supervise learning method which tries to predict the next token. In the Behavior Cloning it instead tries to predict the next expert's action

-  Lines of works
    - **Embodied AI** this researach community in a manipulation task can achieve many skills. But It often ignores the low-level physics like the contact force etc. making it hard to transfer skills to the real env. <br>
    e.g. iGibson 1.0 [Stanford], Habitat 2.0 [Facebook], ThreeDWorld [MIT], Sapien [UCSD], Control Suite [DeepMind], OpenAI Gym [OpenAI], Robo-Manipula-ProcTHOR [AI2], RLbench [Imperial], MetaWorld [UCB], Behavior 1K [Stanford], ManiSkill [UCSD], Issac / Orbit [NVIDIA]
    - **Syntatic data Training** no sim2real gap but highly specific to task it was trained on <br>
    e.g. Dexterous [OpenAI, NVIDIA, MIT], Factory/IndustReaL [NVIDIA], BridgeData [UBC, UPenn, Stanford] 
    - **Training with Real Data** real world robot data comes with huge cost. If you want to do it on a different robot, you need to recollect the data again <br> e.g. Armfarm [Google]
    - **Language Model based** provides a good planning for broad range of tasks but the tasks are often heavily depend on pre-defined skill sets <br>
    e.g. GATO [DeepMind], CLIPort [UW/ NVIDIA], VIMA [NVIDIA], RT1 [Google], PaLM-E [Google], ProgPrompt [NVIDIA], LM for Interactive [MIT]

- **VLM (Visual Language Model)** will be a key success of robot learning, by providing a sufficient large amount of data, it will be robustly deployed on different robots and environments

- Generating real-world data comes with high cost
- Sim2Real zero-shot transfer is important

-  Next Step in robot learning
    1. Generate massive amount of training data
    2. Train large models via supervision and Behavior Cloning
    3. Test in real world environment

- There's an interesting question in the seminar is that "Should we ignore all Physics principles and tries to feed data to a large model as much as possible ? "
