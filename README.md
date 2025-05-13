# Teaching Emotion Through Feature Matching Using CNNs

## Project Overview

This project investigates how Convolutional Neural Networks (CNNs) can be trained to recognize human emotions from facial images, not just by classifying, but by learning the facial features that matter most (like eyes, eyebrows, and mouth positions).

While emotion detection is common, emotional understanding and explainability are still emerging. This project begins that journey by benchmarking deep learning models on facial emotion classification using the FER2013 dataset.

---

## Introduction 

As AI becomes more embedded in everyday aspects, they are also being involved in virtual therapy and mental wellness apps, social robotics and virtual assistants, emotion-aware learning environments. Emotionally intelligent systems are no longer optional, they’re essential.

This project explores whether machines can learn not just to detect emotions, but to explain their reasoning, making human-AI interaction more humane and trustworthy.


## Project Goals

- Train CNN models to classify emotions from facial expressions
- Analyze *which facial features* influence the classification the most
- Begin developing models that can “justify” their predictions through visual explanation tools (e.g., Grad-CAM)
- Evaluate how deep learning performs on low-resolution emotion data, and test alternative approaches when needed

---

## Dataset Used: FER2013

- 48x48 grayscale facial images, converted to RGB
- 7 emotions: `angry`, `disgust`, `fear`, `happy`, `neutral`, `sad`, `surprise`
- Known for being small, noisy, and imbalanced — great for testing robustness
- Split into training, validation, and test sets

---

## Models Tested

| Model             | Training Method         | Weights       | Notes                                    |
|------------------|-------------------------|---------------|------------------------------------------|
| **VGG16**        | From Scratch            | None          | Served as a baseline                     |
| **ResNet50**     | Fine-tuned              | ImageNet      | Poor results, struggled with FER2013     |
| **EfficientNetB0** | Fine-tuned            | ImageNet      | Unexpectedly failed — likely overfit     |
| **Mini-Xception**| In progress             | None          | Lightweight model designed for FER       |

---

## Key Findings (So Far)

- Pretrained models (even EfficientNet) struggled with FER2013
- Imbalanced and low-resolution nature of FER2013 led to poor generalization
- Feature extraction worked, but feature *understanding* still requires improvement
- Class weights, label smoothing, and fine-tuning helped — but not enough

---

## In Progress: Visual Explanation

> Can we show *why* a model thinks someone is angry?

- Grad-CAM and activation maps will be used next
- The goal is to trace CNN attention back to mouth/eye regions to visualize "reasoning"

--- 

## File Structure

