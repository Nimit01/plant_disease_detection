# Plant Disease Detection using MobileNetV2

## Overview

This project is a deep learning based plant disease classification system that identifies plant diseases from leaf images.

The project uses **Transfer Learning with MobileNetV2**, a pretrained convolutional neural network, and adapts it for plant disease classification across **38 different classes**.

## Objective

The main objective is to automatically identify the health condition or disease of a plant from an input leaf image.

The system can be used as the classification component of a plant-health assistance application such as **AgriBuddy**, where the predicted disease can be presented along with a corresponding treatment recommendation.

## Dataset

The project uses a plant leaf image dataset containing **70,295 images belonging to 38 classes**.

The classes include healthy plants as well as diseases affecting:

* Apple
* Blueberry
* Cherry
* Corn (Maize)
* Grape
* Orange
* Peach
* Pepper (Bell)
* Potato
* Raspberry
* Soybean
* Squash
* Strawberry
* Tomato

The complete dataset is not included in this repository because of its size.

## Methodology

### 1. Image Preprocessing

Images are resized to:

```text
224 × 224 pixels
```

and loaded in batches of 32 images.

### 2. Dataset Split

The dataset is divided into:

```text
70% → Training
20% → Validation
10% → Testing
```

The datasets are prefetched using TensorFlow for improved training performance.

### 3. Transfer Learning

**MobileNetV2** pretrained on ImageNet is used as the base model.

The pretrained base layers are initially frozen so that the model can learn plant-disease-specific features using the new classification layers.

### 4. Model Architecture

The classification model consists of:

```text
MobileNetV2
     ↓
Global Average Pooling
     ↓
Dense Layer (256 neurons, ReLU)
     ↓
Dropout (0.3)
     ↓
Output Layer (38 classes, Softmax)
```

This architecture is designed to reuse the feature extraction capabilities of MobileNetV2 while adapting the final layers to the 38 plant disease classes.

### 5. Training

The model uses:

* **Optimizer:** Adam
* **Loss Function:** Categorical Cross-Entropy
* **Metric:** Accuracy
* **Batch Size:** 32

Class weights are also calculated to help handle differences in the number of images among classes.

## Results

One recorded training run achieved:

```text
Test Accuracy: ~84.94%
```

The repository also contains the classification report and confusion matrix generated during model evaluation.

Evaluation includes:

* Precision
* Recall
* F1-score
* Accuracy
* Confusion Matrix

## Prediction

The trained model can be used to classify a new plant leaf image.

The prediction workflow is:

```text
Input Leaf Image
       ↓
Resize Image
       ↓
MobileNetV2 Model
       ↓
Class Probabilities
       ↓
Predicted Plant/Disease Class
       ↓
Treatment Recommendation
```

The `PBl_prediction.ipynb` notebook contains the prediction workflow used with the trained model.

## Repository Structure

```text
plant-disease-detection/
│
├── README.md
├── requirements.txt
├── PBL_70_20_10_split.ipynb
├── PBl_prediction.ipynb
├── plant_disease_model_corrected_50_epochs.h5
│
├── results/
│   ├── classification_report.png
│   └── confusion_matrix.png
│
└── screenshots/
```

## Technologies Used

* Python
* TensorFlow
* Keras
* MobileNetV2
* NumPy
* Matplotlib
* Scikit-learn
* Jupyter Notebook

## How to Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Open the notebooks

```bash
jupyter notebook
```

Open:

```text
PBL_70_20_10_split.ipynb
```

for model training and evaluation.

Open:

```text
PBl_prediction.ipynb
```

for prediction using the trained model.

### 3. Dataset

Download the required plant disease dataset separately and update the dataset path in the training notebook.

## Future Improvements

* Fine-tune more MobileNetV2 layers for improved performance.
* Add data augmentation to improve generalization.
* Expand the number of plant and disease classes.
* Improve the treatment recommendation system.
* Deploy the model as a web or mobile application.
* Evaluate the model on images captured in real-world conditions.

## Disclaimer

This project is intended for educational and experimental purposes. Model predictions should not be treated as a substitute for professional agricultural diagnosis.
