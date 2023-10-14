# CLIP-PACL
### Contrastive Language - Image Pre-training (CLIP) and Patch Aligned Contrastive Learning (PACL)

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![PyTorch 2.0](https://img.shields.io/badge/torch-v2.0-brightgreen)](https://pytorch.org/docs/stable/index.html)
[![PyTorch-Lightning](https://img.shields.io/badge/pytorch_lightning-v2.0.6-orange)](https://lightning.ai/docs/pytorch/latest/)
[![Torchvision 0.15](https://img.shields.io/badge/torchvision-v0.15-green)](https://pytorch.org/vision/stable/index.html)
[![Transformers](https://img.shields.io/badge/Transformers-v4.34.0-lightgreen)](https://huggingface.co/docs/transformers/index)
[![Albumentations](https://img.shields.io/badge/Albumentations-v1.3.1-yellow)](https://albumentations.ai/docs/)

**contents:**

- [Contrastive Language - Image Pre-training](./README.md/#CLIP)
- [Patch Aligned Contrastive Learning](./README.md/#PACL)
- Architectures:

  • [CLIP](./README.md/#clip-architecture)
  • [PACL](./README.md/#pacl-architecture)
  
- [Core Concepts](./README.md/#TERMS)
- [Assignment](./README.md/#Assignment)
- [DEMO](./README.md/#demo)

<h1 align = 'center',id = "CLIP"> 🤗 Contrastive Language - Image Pre-training (CLIP) </h1>

**Intuition:**
The intuition behind OpenAI's CLIP model, as presented in the ["Learning Transferable Visual Models From Natural Language Supervision"](https://colab.research.google.com/corgiredirector?site=https%3A%2F%2Farxiv.org%2Fabs%2F2103.00020) paper, is to create a model that can understand the intricate relationship between natural language descriptions and images. Instead of traditional image classification models that are trained to recognize specific objects or classes (e.g., cars, dogs), CLIP is trained to comprehend complete sentences, which allows it to learn more nuanced patterns and associations between images and text. This broader understanding enables CLIP to perform multiple tasks, including image classification and information retrieval, based on text descriptions.

**Problem:**
The problem that CLIP aims to address is the limitation of conventional computer vision and natural language models that often work in isolation. These models struggle to bridge the gap between images and textual descriptions, making it challenging to perform tasks such as searching for images based on text queries or understanding the context of images. Existing models typically rely on pre-defined labels or categories for images, which can be limiting and fail to capture the richness of human language.

**Solution:**
OpenAI's solution, CLIP, involves training a model to understand the relationships between images and text in a more holistic way. Instead of focusing on specific object recognition or classification, CLIP is trained to associate entire sentences with images. This approach allows CLIP to learn a wide range of connections between different textual descriptions and the visual content they refer to, making it highly versatile. As a result, CLIP can be used for tasks like image classification, object detection, and information retrieval, all while being trained primarily on textual descriptions. This approach breaks the traditional silos between vision and language and enables the model to generalize across various tasks, outperforming specialized models on benchmark datasets, including ImageNet, for image classification.

CLIP or Contrastive Language - Image Pre-training, deviates from the standard practice of fine-tuning a pre-trained model by taking the path of **Zero-Shot Learning**. 

        Zero-shot learning is the ability of the model to perform tasks that it was not explicitly programmed to do. 
    
The core idea of the CLIP paper is essentially to learn visual representation from the massive corpus of natural language data. The paper showed that a simple pre-training task is sufficient to achieve a competitive performance boost in zero-shot learning.

*The objective of the CLIP model can be understood as follows:*

**`Given an image, a set of 32,768 sampled text snippets was paired with it in our dataset. For example, given a task to predict a number from an image, the model is likely to predict that “the number is one” or, “the number is two”, or “the number is XYZ” and so on.`**

<h1 align = 'center',id = "PACL"> 🤗 Patch Aligned Contrastive Learning </h1>

The paper Patch Aligned Contrastive Learning (PACL) introduces, a modified compatibility function for CLIP's contrastive loss which enables the task of Open Vocabulary Semantic Segmentation to be transferred seamlessly without requiring any segmentation annotations during training. Using pre-trained CLIP encoders with PACL, the authors are able to set the state-of-the-art on four different segmentation benchmarks.

- PACL is a modified compatibility function for CLIP's contrastive loss that enables a model to identify regions of an image corresponding to a given text input.
- PACL is applied to pre-trained CLIP encoders to set the state-of-the-art on four different segmentation benchmarks.
- PACL is also applicable to image-level predictions and provides a general improvement in zero-shot classification accuracy compared to CLIP, across a suite of 12 image classification datasets.


<h1 align = 'center', id = "clip-architecture">  🧠 CLIP Architecture </h1>

**Architecture**

<p align='center'>
    <img src='Images/CLIP-Architecture.png' alt='CLIP Architecture' />
    Figure: Summary of the CLIP approach. While standard image models jointly train an image feature extractor and a linear classifier to predict
some label, CLIP jointly trains an image encoder and a text encoder to predict the correct pairings of a batch of (image, text) training
examples. At test time the learned text encoder synthesizes a zero-shot linear classifier by embedding the names or descriptions of the
target dataset’s classes.
</p>

<h1 align = 'center', id = "pacl-architecture">  🧠 PACL Architecture </h1>

**Architecture**

<div style="display: flex; align-items: center; text-align: center;">
    <img src = "Images/PACL-Architecture (2).png" alt='PACL Architecture' style="max-width: 50%; padding: 10px;">
    <div style="flex: 1; text-align: left;">
        <p>Figure: Compatibility function φ(x, y) for Patch Aligned Contrastive Learning (PACL).</p>
        <p>The image encoder f<sub>v</sub> and embedder e^v produce patch-level representations for each image whereas the text encoder ft and embedder et produce the CLS representation for a given text.</p>
        <p>We compute the cosine similarity between the CLS text embedding and the vision patch embeddings and use them as weights to take a weighted sum over vision patch tokens.</p>
        <p>We use the cosine similarity between the weighted sum and the CLS text token as our compatibility φˆ(x, y).</p>
    </div>
</div>
