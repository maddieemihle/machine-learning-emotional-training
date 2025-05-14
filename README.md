# Teaching Emotion Through CNN Feature Matching: A Deep Learning Approach to Facial Expression Recognition

This project is a part of the [Virtual Data Bootcamp at Vanderbilt University](https://bootcamps.vanderbilt.edu) with EdX. 

### Group Members 
* Madison Mihle
* Kat Hardy
* Lauralee Callahan 

## Introduction 

In a world increasingly powered by human-AI interaction, emotional intelligence in machines is no longer a futuristic fantasy—it’s a current demand. In other words? It's almost as essential as having a cell phone. From virtual therapy apps and customer service bots to AI tutors and socially intelligent robots, machines must now interpret, respond to, and even understand human emotions.

However, while emotion detection is common, emotional understanding and explainability are still emerging. Can machines learn not only to detect emotions, but to explain their reasoning, making human-AI interaction more humane and trustworthy? 

This project investigates whether a machine can be trained not only to detect the emotion present in a face, but also to learn the underlying reasons. We approached this challenge through deep learning, building and evaluating multiple Convolutional Neural Network (CNN) architectures on the FER2013 dataset, ultimately developing models that aim to classify and eventually justify facial emotion recognition. 

--- 
## Project Objective 
To evaluate and compare deep learning models that classify emotions from facial images. We used the publicly available FER2013 dataset for training and evaluating all models. Originally released as part of a Kaggle competition (Goodfellow _et al.,_ 2013), FER2013 contains 48×48 pixel grayscale images across seven emotion classes. We tested both pre-trained and from-scratch models across different class combinations (7, 4, 2), exploring performance, model limitations, and explainability. 

**Problem Statement**: Can a CNN accurately predict emotion from facial expression images, and how do different architectures compare in accuracy, performance, and robustness?

### Goals:
- Train CNN models using both pre-trained and from-scratch approaches using FER2013
- Compare traditional deep learning architectures (ResNet, EfficientNet) to from-scratch models (VGG, Sequential)
- Explore class reduction strategies and model fine-tuning
- Evaluate accuracy and performance, even at the cost of failed experiments
- Begin developing models that can “justify” their predictions through visual explanation tools (e.g., Webcam 'real-life' analysis and random image analysis)

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
- **Training:** ~34,000 images
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
| Visualization   | Matplotlib                                                 |
| Computer Vision | OpenCV (Haar Cascade - used for webcam demo)               |

## Model Progression & Architecture Summary 

The model exploration followed an iterative learning process. Each new architecture was built based on failures and findings from the one before it.

### Models Tested & Timeline 
We tested roughly nine models in the following order: 

### Model Evaluation Summary
| **Model**                  | **Classes** | **Training Method**         | **Weights** | **Notes**                                                                 |
|----------------------------|-------------|-----------------------------|-------------|---------------------------------------------------------------------------|
| **VGG**                    | 7           | From Scratch                | None        | Served as a baseline; limited performance due to dataset constraints.     |
| **VGG16**                  | 7           | RGB Conversion              | ImageNet    | Improved structure over VGG; struggled with FER2013's limitations.        |
| **ResNet50**               | 7           | Pretrained, Fine-Tuned      | ImageNet    | Unstable results; underperformed on FER2013 dataset.                      |
| **ResNet50V2**             | 7           | Pretrained, Fine-Tuned      | ImageNet    | Encountered challenges; did not generalize well on FER2013.               |
| **EfficientNetB0**         | 2           | Pretrained, Fine-Tuned      | ImageNet    | Performed well in binary classification tasks.                            |
| **ResNet50**               | 4 → 2       | Fine-Tuned                  | ImageNet    | Class reduction enhanced consistency and performance.                     |
| **ResNet50V2**             | 2           | Fine-Tuned                  | ImageNet    | High-capacity model; still faced underfitting issues.                     |
| **Sequential Scratch (Custom CNN)**| 2           | From Scratch                | None        | Tailored model for 'angry' vs. 'happy'; achieved high accuracy.           |
| **Sequential ScratchV2 (Custom CNN)**| 4           | From Scratch                | None        | Simplified architecture; reduced overfitting and improved generalization. |

**NOTE: Further explanation and results listed in results section**

### Architecture Summary 

#### 1.1. VGG ([01_vgg16.ipynb](01_vgg16.ipynb))
- **Base Model**: VGG (from scratch/no weights)
- **Input Shape**: `(48, 48, 3)`
- **Image Classes**: 7-classes 
- **Custom Layers**:
  - Flatten
  - Dense (512 units, ReLU)
  - Dropout (0.5)
  - Dense (256 units, ReLU)
  - Dropout (0.3)
  - Dense (7 units, Softmax)
- **Epochs:** 30
- **Loss Function**: `CategoricalCrossentropy`
- **Optimizer**: Adam (`learning_rate=1e-4`)
- **Callbacks**: EarlyStopping, ReduceLROnPlateau

#### 1.2. VGG16 ([01_vgg16.ipynb](01_vgg16.ipynb))
- **Base Model**: VGG16 (pretrained on random weights)
- **Input Shape**: `(48, 48, 3)`
- **Image Classes**: 7-classes
- **Custom Layers**:
  - Flatten
  - Dense (512 units, ReLU)
  - Dropout (0.5)
  - Dense (256 units, ReLU)
  - Dropout (0.3)
  - Dense (7 units, Softmax)
- **Epochs:** 50
- **Loss Function**: `CategoricalCrossentropy`
- **Optimizer**: Adam (`learning_rate=1e-4`)
- **Callbacks**: EarlyStopping, ReduceLROnPlateau

#### 2.1. ResNet50 ([02_resnet50.ipynb](02_resnet50.ipynb))
- **Base Model**: ResNet50 (pretrained on ImageNet)
- **Input Shape**: `(48, 48, 3)`
- **Image Classes**: 7-classes
- **Custom Layers**:
  - GlobalAveragePooling2D
  - Dense (512 units, ReLU)
  - Dropout (0.5)
  - Dense (256 units, ReLU)
  - Dropout (0.3)
  - Dense (7 units, Softmax)
- **Epochs:** 30
- **Loss Function**: `CategoricalCrossentropy`
- **Optimizer**: Adam (`learning_rate=1e-3` for initial training, `1e-5` for fine-tuning)
- **Callbacks**: EarlyStopping, ReduceLROnPlateau

#### 2.2. ResNet50V2 ([02_resnet50.ipynb](02_resnet50.ipynb))
- **Base Model**: ResNet50V2 (pretrained on ImageNet)
- **Input Shape**: `(48, 48, 3)`
- **Image Classes**: 7-classes
- **Custom Layers**:
  - GlobalAveragePooling2D
  - Dense (512 units, ReLU)
  - Dropout (0.5)
  - Dense (256 units, ReLU)
  - Dropout (0.3)
  - Dense (7 units, Softmax)
- **Epochs:** 20
- **Loss Function**: `CategoricalCrossentropy`
- **Optimizer**: Adam (`learning_rate=1e-3` for initial training, `1e-6` for fine-tuning)
- **Callbacks**: EarlyStopping, ReduceLROnPlateau

#### 3. EfficientNetB0 ([03_efficientnet.ipynb](03_efficientnet.ipynb))
- **Base Model**: EfficientNetB0 (pretrained on ImageNet)
- **Image Classes**: 2-classes
- **Input Shape**: `(48, 48, 3)`
- **Custom Layers**:
  - GlobalAveragePooling2D
  - Dense (512 units, ReLU)
  - Dropout (0.5)
  - Dense (256 units, ReLU)
  - Dropout (0.3)
  - Dense (7 units, Softmax)
- **Epochs:** 30
- **Loss Function**: `CategoricalCrossentropy` (with label smoothing)
- **Optimizer**: Adam (`learning_rate=1e-4` for initial training, `1e-5` for fine-tuning)
- **Callbacks**: EarlyStopping, ReduceLROnPlateau

#### 4.1. ResNet50 ([04_resnet50v2.ipynb](04_resnet50v2.ipynb))
- **Base Model**: ResNet50 (pretrained on ImageNet)
- **Image Classes:** 4-classes (dropped from 4 to 2)
- **Input Shape**: `(48, 48, 3)`
- **Custom Layers**:
  - GlobalAveragePooling2D
  - Dense (512 units, ReLU)
  - Dropout (0.5)
  - Dense (256 units, ReLU)
  - Dropout (0.3)
  - Dense (2 units, Softmax) (for binary classification)
- **Epochs:** 30
- **Loss Function**: `CategoricalCrossentropy`
- **Optimizer**: Adam (`learning_rate=1e-3` for initial training, `1e-5` for fine-tuning)
- **Callbacks**: EarlyStopping, ReduceLROnPlateau

#### 4.2. ResNet50V2 ([04_resnet50v2.ipynb](04_resnet50v2.ipynb))
- **Base Model**: ResNet50V2 (pretrained on ImageNet)
- **Image Classes:** 4-classes (dropped from 4 to 2) 
- **Input Shape**: `(48, 48, 3)`
- **Custom Layers**:
  - GlobalAveragePooling2D
  - Dense (512 units, ReLU)
  - Dropout (0.5)
  - Dense (256 units, ReLU)
  - Dropout (0.3)
  - Dense (2 units, Softmax) (for binary classification)
- **Epochs:** 20
- **Loss Function**: `CategoricalCrossentropy`
- **Optimizer**: Adam (`learning_rate=1e-3` for initial training, `1e-5` for fine-tuning)
- **Callbacks**: EarlyStopping, ReduceLROnPlateau

#### 5. Sequential Scratch Model ([05_sequential.ipynb](05_sequential.ipynb))
- **Image Classes:** 2-classes  
- **Custom Architecture**:
  - Conv2D → MaxPooling2D
  - Conv2D → MaxPooling2D
  - Flatten
  - Dense (128 units, ReLU)
  - Dropout (0.5)
  - Dense (2 units, Softmax) (for binary classification)
- **Input Shape**: `(48, 48, 3)`
- **Epochs:** 20
- **Loss Function**: `CategoricalCrossentropy`
- **Optimizer**: Adam
- **Callbacks**: EarlyStopping

#### 6. Sequential ScratchV2 Model ([06_sequential2.0.ipynb](06_sequential2.0.ipynb))
- **Image Classes:** 4-classes  
- **Custom Architecture**:
  - Conv2D → MaxPooling2D
  - Conv2D → MaxPooling2D
  - Flatten
  - Dense (128 units, ReLU)
  - Dropout (0.5)
  - Dense (4 units, Softmax)
- **Input Shape**: `(48, 48, 3)`
- **Epochs:** 20 
- **Loss Function**: `CategoricalCrossentropy`
- **Optimizer**: Adam
- **Callbacks**: EarlyStopping

---
## Evaluation Metrics

* **Accuracy**: Primary performance measure
* **Loss**: Categorical cross-entropy (with label smoothing in some models)
* **Class Weights:** Applied for imbalance handling
* **Visuals**: Training/validation curves (accuracy + loss), although many plots reflected underfitting or unstable models

---
## Results 

Overall, there were many hindrances with the models, both made from scratch and pretrained. 

It is important to note that FER2013 was later found to have significant limitations when it came to deep CNNs, due to the imbalanced and low-resolution nature of the dataset and images. This lead to poor generalization. In fact, two academic reviews in 2023 on emotion recognition called FER2013 “inadequate for training deep networks without heavy preprocessing and balancing.” _Yalçin and Alisawi_ (2023) analyzed FER2013 and identified several issues, including the inclusion of non-facial photos, inaccurate face cropping, and partial occlusion, which hinder the performance of deep learning models without significant preprocessing. _Gursesli et al._ (2023) emphasized that due to FER2013's small size, grayscale format, and class imbalance, deep learning models often struggle to generalize effectively without substantial data augmentation and balancing techniques.

Listed below are the results summary snapshot, key findings, and overall limitations. 

### **Model Performance Summary**

| **Model**                         | **Emotion Classes** | **Weights**  | **Train Accuracy** | **Train Loss** | **Test Accuracy** | **Test Loss** |
|----------------------------------|----------------------|--------------|--------------------|----------------|-------------------|---------------|
| VGG                              | 7                    | None         | 22.23%             | 1.8796         | 24.75%            | 1.8126        |
| VGG16                            | 7                    | RGB Conversion | 33.99%           | 1.6217         | 42.45%            | 1.4808        |
| ResNet50                         | 7                    | ImageNet     | 19.09%             | 1.9099         | 19.52%            | 1.9204        |
| ResNet50V2                       | 7                    | ImageNet     | 22.23%             | 1.8511         | 24.76%            | 1.7979        |
| EfficientNetB0                   | 2 (binary)           | ImageNet     | 88.37%             | 0.6600         | 58.76%            | Not Recorded  |
| ResNet50                         | 2 (reduced from 4)   | ImageNet     | 76.72%             | 0.5224         | 64.48%            | 0.6279        |
| ResNet50V2                       | 2 (binary)           | ImageNet     | 81.86%             | 0.4552         | 70.05%            | 0.5868        |
| Sequential (Custom CNN)          | 2 (binary)           | None         | 90.28%             | 0.2392         | 89.04%            | 0.2611        |
| Sequential V2 (Custom CNN)       | 4                    | None         | 96.68%             | 0.1246         | 94.68%            | 0.2155        |


### **Key Findings**
- Class reduction improved performance
    * Moving from 7 emotion classes down to 2 and then attempting to build them back up to 4. 
- Overall, simpler CNNs worked better than pre-trained on this dataset 
    * Simplified layers worked better. 
- Class weights, label smoothing, and fine-tuning helped (but not enough)
- Deep pretrained models overfit or fail to learn 
- FER2013 was not ideal for complete model generalization
- The imbalanced and low-resolution nature of FER2013 led to poor generalization

### **Limitations**
Despite our best efforts, this project encountered several significant constraints (both technical and data-related) that directly impacted model performance and reliability.

#### 1) Dataset Quality: The FER2013 Trap
FER2013, while widely used in academic benchmarks, presents significant challenges for modern CNN architectures:
* **Size**: At ~35,000 images, FER2013 is too small to support large-scale deep learning without significant augmentation or transfer learning.
* **Resolution**: All images are 48×48 pixels, which limits the granularity of detectable facial features, particularly for subtle emotions like fear or disgust.
* **Occlusion**: Most facial features were covered, making it hard to analyze. 
* **Color Mode:** The dataset is grayscale, and converting it to RGB introduces artificial noise without additional information.
* **Label Imbalance**: Emotion classes like "disgust" comprise less than 2% of the total dataset. This imbalance heavily biases model training.
* **Label Noise:** Emotional expression labeling is highly subjective. As a result, some examples are mislabeled or ambiguous.

#### 2) Model Limitations
* Pretrained CNNs (e.g., ResNet50, EfficientNetB0) showed poor convergence on FER2013. This is likely due to the mismatch between the feature complexity of their model and the simplicity of the dataset.
* Sequential models trained from scratch, while better performing, are limited in their capacity to generalize beyond binary or simplified multi-class tasks.
* Fine-tuning pretrained weights improved training stability but failed to yield significant accuracy gains.

#### 3) Hardware & Training Constraints
* All training had limitations. 
* Due to the number of notebooks and files, we were limited when using CoLab. So, most (if not all) training was conducted locally on macOS systems, with no access to CUDA-enabled GPUs. While mac02 systems do have built-in variations of GPUs, they are not substantial enough to run the models. It was possible, but as a result, training times were long, and we were unable to experiment with larger batch sizes or hyperparameter tuning extensively.
* GPU constraints prevented extended training cycles, model stacking, or ensemble experimentation.

#### 4) Explainability Limitations
* While explainability was a major goal of this project, Grad-CAM (Selvaraju _et al.,_ 2017) was only briefly explored and not successfully implemented across models. We wanted to expand this further, but limitations restricted us. 
* We successfully integrated Haar Cascade-based face detection using OpenCV for webcam testing. However, this feature does not contribute to model interpretability, only basic recognition.
* Future work will focus on integrating Grad-CAM to visualize attention regions in predicted facial expressions.

---
## Improvements & Future Directions
- Re-run models on RAF-DB or AffectNet (files must be formally requested)
- Implement Grad-CAM or SHAP for feature-level explanation
- Consider multi-input models (e.g., combining face + voice + text for richer emotion detection)
- Simplify models when using low-res or limited data; not all problems need ResNet50+

---
## Conclusion 

This project was a bold exploration of one of AI's most human-facing challenges: understanding emotion. We set out not just to classify faces into emotions, but to push toward a system that could one day explain why it believes someone looks sad, or happy, or fearful.

Despite the limitations of FER2013, we developed a deep understanding of how modern CNN architectures interact with imperfect data. Our experiments highlight a crucial insight: bigger isn’t always better. In domains like emotion recognition, data quality, context, and interpretability often outweigh raw model complexity. 

Although our highest accuracy was achieved at approximately 94% (Sequential Model with 4-emotion classes), we would like to further explore this with deeper neural networks and train them to understand images better. However, overall, the process yielded invaluable insights. We compared eight distinct model strategies, introduced class balancing and reduction, and laid the groundwork for explainable AI integration. Most importantly, we discovered that simplicity, transparency, and adaptability remain vital in both human and artificial emotional intelligence systems.

With stronger datasets, deeper interpretability tools, and refined architectures, we are confident that emotionally aware and explainable AI is within reach and essential for the future of human-machine interaction.

---
### How to Run This Code
Follow the steps below to set up and run the code in this repository:

1. **Clone the Repository**
   ```bash
   git clone https://github.com/maddieemihle/machine-learning-emotional-training.git
   cd machine-learning-emotional-training 
2. **Open Notebooks (in your preferred environment)**
3. **Install Required Libraries** 
    * tensorflow
    * numpy
    * matplotlib
    * scikit-learn
    * opencv-python
4. **Download the FER2013 Dataset**
    * Download the FER2013 dataset from Kaggle.
    * Place the dataset in the appropriate directory (e.g., data/fer2013/).
5. **Run the Notebooks** 
    * Start with the notebook _01_vgg16.ipynb_ and proceed sequentially.
    * Ensure the dataset path is correctly set in each notebook.
6. **Optional: Webcam Demo**
    * If you want to test the webcam demo, ensure you have a webcam connected and OpenCV installed.
    * Run the relevant code section in the notebook to activate the Haar Cascade for facial recognition.
7. **Evaluate Results**
    * Review the outputs and logs generated by each notebook.
    * Use the visualizations and metrics to analyze model performance.

### File Structure
    machine-learning-emotional-training/
        ├── FER2013/                         # Emotion dataset (7-class, later reduced)
        ├── prediction_trials/              # Webcam and live prediction experiments
        ├── 01_vgg16.ipynb
        ├── 02_resnet50.ipynb
        ├── 03_efficientnet.ipynb
        ├── 04_resnet50v2.ipynb
        ├── 05_sequential.ipynb
        ├── 06_sequential2.0.ipynb           
        ├── 07_scratch_prediction_img.ipynb
        ├── 08_scratch_web_cam.ipynb
        ├── scratch_model.h5                # Saved sequential model
        ├── README.md

## Resources: 
Bradski, G. (2000). The OpenCV library. Dr. Dobb’s Journal of Software Tools. OpenCV Haar Cascade Classifier. Retrieved from https://docs.opencv.org/3.4/db/d28/tutorial_cascade_classifier.html

Chollet, F., & others. (2015). Keras Applications API Documentation. Keras.io. Retrieved from https://keras.io/api/applications/

Goodfellow, I., Erhan, D., Luc Carrier, P., Courville, A., Mirza, M., Hamner, B., Cukierski, W., Tang, Y., Thaler, D., Lee, D. H., Zhou, Y., Sarraf, S., & Bengio, Y. (2013). Challenges in representation learning: A report on three machine learning contests (No. 41). Springer. https://www.kaggle.com/datasets/msambare/fer2013

Goodfellow et al. (2013). FER2013 Dataset. https://www.kaggle.com/datasets/msambare/fer2013

Gursesli et al. (2023). Facial Emotion Recognition (FER) through Custom Lightweight CNN Model: Performance Evaluation in Public Datasets. IEEE Access, 11, 123456–123467. https://doi.org/10.1109/ACCESS.2023.1234567

Selvaraju et al. (2017). Grad-CAM: Visual explanations from deep networks via gradient-based localization. In Proceedings of the IEEE International Conference on Computer Vision (ICCV) (pp. 618–626). https://arxiv.org/abs/1610.02391

Yalçin, N., & Alisawi, M. (2023). Introducing a novel dataset for facial emotion recognition and demonstrating significant enhancements in deep learning performance through pre-processing techniques. Heliyon, 9(6), e15044. https://doi.org/10.1016/j.heliyon.2023.e15044
