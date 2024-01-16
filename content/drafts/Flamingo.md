---
author: "trapoom555"
title: "[Draft] Flamingo"
date: "2024-01-15"
math: mathjax
draft: true
---
# Notes

- It is a Visual Language Model (VLM) with few-shots capability
- Contributions
    - Bridge powerful Vision-only and Language-only model
    - Can handle with arbitrarily interleaved sequence of text and image
    - Seamlessly ingest images or videos as input

- Flamingo was trained on a large-scale web copra (*without data annotation*) because of its arbitrarily interleaved flexibility
- Flamingo produce free-form text as an output
- It outperform fine-tuning model by just few-shot learning on 6 tasks out of 16 tasks and outperform *all* zero-shot or few-shot methods in 16 benchmarks

> Because of the large-scale data, it shows a *in-context few-shot learning capabilities* and Unlike Frozen, **It can achieve a new state of the art of many tasks**

- Architecture
    - Visual Processing
        - Vision Encoder : NF-ResNet pretrained with contrastive objective with image and text pairs and is frozen in Flamingo
        - Perceiver Resampler : Recieve spatio-temperal features from Vision Encoder and output a fixed number of Visual Tokens (64) to reduce computational complexity
    - Language Models : Chinchilla models of 1.4B, 7B, 70B and this will be combined to the Vision Encoder and other components in the model and will produce Flamingo 1.4B, 7B
        - It is Frozen to prevent "catastrophic forgetting", the ablation study shows that when fine-tuning the LLM layer the performance dropped by 8%
    - Fusing Visual and Text information
        - GATED XATTN-DENSE : It's a new layer which are added between the frozen pre-trained language model layers. The inner architecture is quite similar to the original Transformer block, using Text information as a query and Visual information (connected from the Perceiver Resampler) as a key and values. The only difference to the original Transformer block is that the $tanh(\alpha)$ gated is multiplied to the output of sublayer to ensure that at initialization, the conditioned model yields the same results as the original language model. $\alpha = 0$ at initialization which makes $tanh(\alpha) = 0$ and learnable so that, at the beginning, the results of the conditioned model should be equal to the results from only LLM. By adding this gate it improves the model's training stability and performance.

- Dataset
    - M3W : MultiModal MassiveWeb dataset (self collected by Deepmind), 43 million webpages. Sample token length of 256 tokens and take up to 5 images
    - Image/Video and text pairs : leverage ALIGN dataset (1.8 B) and added self collected dataset, 312 mission image text pairs and 26 million short videos text pairs.
- Training
    - Try to model the Likelihood
    $y$ is texts and $x$ is images and videos
    $$p(y|x) = \prod_{l=1}^L p(y_l|y_{<l}, x_{\le l})$$
    - Video sampling : 1 FPS
    - Per-image / Video Masking : In the cross attention layer, just one previous image is attended directly with the text tokens. The rest previous images are masked. But the information of all previous images are still kept in the self-attention layer. This helps the model to generalize the number of visual input in the sequence. (experimentally supported)
    - Objective : Minimizing a weighted sum of per-dataset expected negative log-likelihoods

    $$\sum_{m=1}^{M} \lambda_m \cdot \mathbb{E}_{(x, y)\sim \mathcal{D}_m} \left[ - \sum_{l=1}^L \log p(y_l | y_{<l}, x_{\le l})\right]$$

- Evaluation
    The goal is to build the model that can rapidly adapt to diverse and challenging tasks.
    - Visual question-answering
    - Captioning Task
    - Close-ended task such as multiple choice visual question answering

- Limitations
    - Inherit weekness of the LLM
        - hallucination
        - LLMs generalise poorly to sequences longer than the training ones
    - the classification performance of Flamingo lags behind the SOTA of contrastive models
    - In-context learning is sensitive and costly at inference. -> poor scalability

