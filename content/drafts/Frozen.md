---
author: "trapoom555"
title: "[Draft] Frozen LLM"
date: "2024-01-15"
math: mathjax
draft: true
---

# Notes

- The knowledge from LLM can transfer to the non-linguistic tasks.
- LLM few-shot transfer learning to multimodal settings
- few-shot learner : The LLM can learn new task by prompting. No gradient updates.
- **Frozen** : the method allows LLM to access visual information without updating weights. It adds a Vision Encoder which is trained to transform image in the word token space. Vision Encoder's weight will be updated during training with (text, image) pairs
- It shows capability of 0-shot VQA, 1-shot outside knowledge VQA and few-shot image classification.
- Their Vision Encoder is NF-ResNet50
- visual prefix
Suppose embedding dimension $D$
*Linear* transform an output of the Vision Encoder to be $D\times n$ and call it as a Visual Prefix

- It was trained on 3M image-caption dataset.
> The experiment shows that fine-tuning the LLM hurts the generalization since there're not many text-caption data compaired to the text-only data

- **Multimodal Few-shot learner** 
    - rapid adaption to new tasks
        - Frozen perform best on new tasks compared to 3 baselines
            - train end-to-end from scratch
            - also finetuned LLM
            - blind : doesn't consider visual input
        - Using larger LLM does affect the performance
    - Fast access to general knowledge
        - Wright example
    - Fast binding of visual and linguistic elements
        - few shot bind new concept of image and text together

- Limitation : the few-shot method is still far from state of the art of compared to systems that use the full training set for those tasks.

