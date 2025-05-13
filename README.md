# Teaching Emotion Through CNN Feature Matching: A Deep Learning Approach to Facial Expression Recognition

This project is a part of the [Virtual Data Bootcamp at Vanderbilt University](https://bootcamps.vanderbilt.edu) with EdX. 

### Group Members 
* Madison Mihle
* Kat Hardy
* Lauralee Calahan 

---
## Introduction 

In a world increasingly powered by human-AI interaction, emotional intelligence in machines is no longer a futuristic fantasy-—it’s a current demand. In other words? It's almost as essential as having a cell phone. From virtual therapy apps and customer service bots to AI tutors and socially intelligent robots, machines must now interpret, respond to, and even understand human emotions.

However, while emotion detection is common, emotional understanding and explainability is still emerging. Can machines learn not only to detect emotions, but to explain their reasoning, making human-AI interaction more humane and trustworthy? 

This project explores whether a machine can be taught not just to detect what emotion is present in a face, but to learn why. We approached this challenge through deep learning, building and evaluating multiple Convolutional Neural Network (CNN) architectures on the FER2013 dataset, ultimately developing models that aim to classify and eventually justify facial emotion recognition. 


The objective of this project is to investigate how Convolutional Neural Networks (CNNs) can be trained to recognize human emotions from standard facial images. Not just by classifying, but by learning the facial features that matter most (like eyes, eyebrows, and mouth positions). Ultimately, this project explores whether machines can learn not only to detect emotions, but to explain their reasoning, making human-AI interaction more humane and trustworthy.

--- 
## Project Objective 
To evaluate and compare deep learning models that classify emotions from facial images, using the FER2013 dataset as a base. We tested both pre-trained and from-scratch models across different class combinations (7, 4, 2), exploring performance, model limitations, and explainability.

**Problem Statement**: Can a CNN accurately predict emotion from facial expression images, and how do different architectures compare in accuracy, performance, and robustness?

### Goals:
- Train CNN models using both pre-trained and from-scratch approaches using FER2013
- Compare traditional deep learning architectures (ResNet, EfficientNet) to from-scratch models (VGG, Sequential)
- Explore class reduction strategies and model fine-tuning
- Evaluate accuracy and performance, even at the cost of failed experiments
- Begin developing models that can “justify” their predictions through visual explanation tools (e.g., Webcam 'real-life' analysis and random image anaylsis)

---
## Dataset Description: 
- **Name:** [FER2013](https://www.kaggle.com/datasets/msambare/fer2013)
- **Format:** 48x48 grayscale images (converted to RGB in some models), CSV or directory-based with train/test folders
- **Emotional Classes:** 7 original labels
    * `angry`
    * `disgust`
    * `fear`
    * `happy`
    * `neutral`
    * `sad`
    * `surprise`
- **Preprocessing:** Normalization, RGB conversion, class reduction 
- **Training:** ~24,000 images
- **Validation:** ~4,000 images
- **Test:** ~7,000 images
- **Processing:** Class reduction (from 7 → 4 → 2 in later models)

---

## Technologies & Tools Used 

| Category        | Tools/Libraries                                            |
| --------------- | ---------------------------------------------------------- |
| Language        | Python 3.11                                                |
| Deep Learning   | TensorFlow, Keras, Scikit-learn                            |
| Data Processing | Pandas, NumPy                                              |
| IDE / Platforms | Visual Studio Code (VSCode), Jupyter Notebooks             |
| Visualization   | Matplotlib, OpenCV (Haar Cascade - used for webcam demo)   |

## Model Progression & Architecture Summary 

The model exploration folled an iterative learning process. Each new architecture was built based on failures and findings from the one before it.

### Models Tested & Timeline 
We tested roughly 7 models in the following order: 

| Model            | Training Method         | Weights       | Notes                                    |
|------------------|-------------------------|---------------|------------------------------------------|
| **VGG**          | From Scratch            | None          | Served as a baseline                     |
| **VGG16**        | Pretrained              | None          | Better structure, but struggled with FER2013 |
| **ResNet50**     | Pretrained, Fine-Tuned  | ImageNet      | Poor results, unstable                   |
| **EfficientNetB0** | Pretrained, Fine-Tuned | ImageNet     | Unexpectedly failed — likely overfit     |
| **Mini-Xception**| From Scratch            | None          | Custom for FER2013, lightweight          |
| **ResNet50**     | Fine-Tuned (4-class)    | ????          | Class reduction improved consistency     |  
| **ResNet50V2**   | Fine-Tuned (2-class)    | ????          | High-capacity model, still underfit      |  
| **Squential**    | From Scatch (2-class)   | None          | Custom model trained on angry vs. happy only |  
| **Squential**    | From Scatch (4-class)   | None          | Simplified architecture, reduced overfitting |  

### Architecture Summary 

#### 1. VGG16 ([01_vgg16.ipynb](01_vgg16.ipynb))
- **Base Model**: VGG16 (pretrained on ImageNet or random weights)
- **Input Shape**: `(48, 48, 3)`
- **Custom Layers**:
  - Flatten
  - Dense (512 units, ReLU)
  - Dropout (0.5)
  - Dense (256 units, ReLU)
  - Dropout (0.3)
  - Dense (7 units, Softmax)
- **Loss Function**: `CategoricalCrossentropy`
- **Optimizer**: Adam (`learning_rate=1e-4`)
- **Callbacks**: EarlyStopping, ReduceLROnPlateau

#### 2. ResNet50 ([02_resnet50.ipynb](02_resnet50.ipynb))
- **Base Model**: ResNet50 (pretrained on ImageNet)
- **Input Shape**: `(48, 48, 3)`
- **Custom Layers**:
  - GlobalAveragePooling2D
  - Dense (512 units, ReLU)
  - Dropout (0.5)
  - Dense (256 units, ReLU)
  - Dropout (0.3)
  - Dense (7 units, Softmax)
- **Loss Function**: `CategoricalCrossentropy`
- **Optimizer**: Adam (`learning_rate=1e-3` for initial training, `1e-5` for fine-tuning)
- **Callbacks**: EarlyStopping, ReduceLROnPlateau
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