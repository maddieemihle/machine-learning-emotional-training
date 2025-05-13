# Teaching Emotion Through Feature Matching Using CNNs

## Project Overview

This project investigates how Convolutional Neural Networks (CNNs) can be trained to recognize human emotions from facial images, not just by classifying, but by learning the facial features that matter most (like eyes, eyebrows, and mouth positions).

While emotion detection is common, emotional understanding and explainability are still emerging. This project begins that journey by benchmarking deep learning models on facial emotion classification using the FER2013 dataset.

## Introduction 

As AI becomes more embedded in everyday aspects, they are also being involved in virtual therapy and mental wellness apps, social robotics and virtual assistants, emotion-aware learning environments. Emotionally intelligent systems are no longer optional, they’re essential.

This project explores whether machines can learn not just to detect emotions, but to explain their reasoning, making human-AI interaction more humane and trustworthy.


## Project Goals

- Train CNN models to classify emotions from facial expressions
- Analyze *which facial features* influence the classification the most
- Begin developing models that can “justify” their predictions through visual explanation tools (e.g., Grad-CAM)
- Evaluate how deep learning performs on low-resolution emotion data, and test alternative approaches when needed

---

## Process 

*here is where we will put a step by step walk through of the code and how it was formulated for each of the models and the webcam visualizations and image visualizations.*

## Dataset Used: FER2013

[FER2013](https://www.kaggle.com/datasets/msambare/fer2013)

- Format: 48x48 grayscale images (converted to RGB)
- Categories: `angry`, `disgust`, `fear`, `happy`, `neutral`, `sad`, `surprise`
- Training: ~24,000 images
- Validation: ~4,000 images
- Test: ~7,000 images

---

## Models Tested

| Model             | Training Method         | Weights       | Notes                                    |
|------------------|-------------------------|---------------|------------------------------------------|
| **VGG16**        | From Scratch            | None          | Served as a baseline                     |
| **ResNet50**     | Fine-tuned              | ImageNet      | Poor results, struggled with FER2013     |
| **EfficientNetB0** | Fine-tuned            | ImageNet      | Unexpectedly failed — likely overfit     |
| **Mini-Xception**| From Scratch            | None          | Custom for FER2013, lightweight          |
| **ResNet50**     | Fine-tuned              | ????          | Dropped from 7 to 4 classes              |  
| **ResNet50V2**   | Fine-tuned              | ????          | Only 2 classes                           |  
| **Squential**    | From Scatch             | None          | 2 classes                                |  
| **Squential**    | From Scatch             | None          | 4 classes                                |  
---

## Results 

### Results Summary (FER2013)

| Model            | Final Accuracy | Loss      | Observations                          |
|------------------|----------------|-----------|----------------------------------------|
| VGG16            | ~43%           | 1.49      | Underfits, struggles with fear/disgust |
| ResNet50 (FT)    | ~25–27%        | 1.9+      | Did not generalize well                |
| EfficientNetB0 (FT) | ~1.5% (!)   | 1.93+     | Unexpected failure — likely overfitting |
| Mini-Xception    | ??             | ???        | Better suited for FER2013              |
| ResNet50 (4)     | ??             | ???        | ?????                                  |
| ResNet50V2 (2)   | ??             | ???        | ?????                                  |
| Sequential (2)   | ??             | ???        | ?????                                  |
| Sequential (4)   | ??             | ???        | ?????                                  |

* Note: Even pretrained models performed poorly, reinforcing FER2013's limitations as a training set for deep CNNs.

### Key Findings

- Pretrained models (even EfficientNet) struggled with FER2013
- Imbalanced and low-resolution nature of FER2013 led to poor generalization
- Feature extraction worked, but feature *understanding* still requires improvement
- Class weights, label smoothing, and fine-tuning helped — but not enough

---


## Conclusion 


** need to note that the FER2013 was in greyscale and I had to convert to rgb. The more complex models (i.e. ResNet50, ResNet50V2, EfficientNetB0) need to be in color (RGB). Also they have to have a three-channel and when I tried to change from 48, 48, 3 --> 1, the code would not run.


### Tools and Technologies Used
- Haar Cascade (this is the webcame pre-existing library)



### File Structure



## Resources: 