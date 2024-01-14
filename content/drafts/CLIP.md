---
author: "trapoom555"
title: "[Draft] CLIP"
date: "2024-01-14"
math: mathjax
draft: true
---

# Notes
- Before CLIP, CV models are not robust. Giving test data which are not in an imageNet, accuracy drop significantly.
- It is doubted that the old CV models has low **scalability** which needs more training data (which is labor intensive and costly) to generalize well
- CLIP can maintain an accuracy while testing on different dataset
- **zero-shot** performance on a great variety of image classification datasets
- **zero-shot**  to nearly arbitrary visual classification tasks.

- CLIP learns from 400M (text, image) pairs on the internet. While imagenet is very costly (14M images, 25K workers)

- Traditional CV model doesn't generalize well, when practitioner needs to apply to the task with their new dataset, they need to fine-tune it.

- But with CLIP, just tell the Text encoder about the concept what you want to classify

- Unlike many models that are cheating on the benchmark, CLIP tested a performance on unseen data

- Limitaiton: 
1. CLIP fails to do more complex task than recognizing common object. for example, counting people in the image. 
2. It sensitives to wording -> need prompt engineering
3. CLIP also still has poor generalization to images not covered in its pre-training dataset. For instance, although CLIP learns a capable OCR system, when evaluated on handwritten digits from the MNIST dataset, zero-shot CLIP only achieves 88% accuracy, well below the 99.75% of humans on the dataset.

## Contrastive Learning

Is a concept that learning embedding space which similar pairs are close to each other while dissimilar pairs are far apart.

## Approach
- Given a batch $N$ of (image, text) pairs, CLIP will compute the cosine similarity matrix of $N \times N$ possible pairs
- Jointly train both Image and Text encoder to maximize cosine similarity of $N$ real pairs while minimizing $N^2 -N$ incorrect pairing
