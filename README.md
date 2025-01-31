
# AI-image-classifier
## Artificial Intelligence Generated Images VS. Human Generated Images

<p align="center">
   
   ![blue-globe-banner](https://github.com/user-attachments/assets/f1887d8c-e69f-4cf6-b431-be71803e623b)

   -----------------------------------------

|     Image Authenticity Issue         |     Model Marketed For All Users          |  Current Need For Detection      |
|--------------------------------------|:-----------------------------------------:|---------------------------------:|
|     News articles with images        |     User inputs image in question         | Preserving journalism            |
|        Crime scene images            |  Identify level of image manipulation     | Forensic accuracy                 |
|        Social Media Images           | Easy image input for all levels of users  | Cultural Authenticity            |
|        Scientific Reports            |      Instant results for the user         | Stability of the Sciences        |
|     Marketing and Advertising        |      Applicable on all platforms          | Consumer Protect (FTC)           |
|     Campaign or promotion images     | Easy to share results for fact checking   | Fair and Just campaigns          |

--------------------------------------------


"The dataset consists of authentic images sampled from the Shutterstock platform across various categories, including a balanced selection where one-third of the images feature humans. These authentic images are paired with their equivalents generated using state-of-the-art generative models. This structured pairing enables a direct comparison between real and AI-generated content, providing a robust foundation for developing and evaluating image authenticity detection systems." (Kaggle)

https://www.kaggle.com/datasets/alessandrasala79/ai-vs-human-generated-dataset

### **Goal**

Our goal is to develop a model that classifies images as AI generated or 'real' using the dataset of 40k AI and 40k real.
This model that will help differentiate between human and machine created visuals. 

## Files Overview

- **`ai_image_classifier_data_cleanup.ipynb`**  
  Preprocesses images by resizing, normalizing, converting grayscale to RGB, augmenting data, and preparing it for model training.

- **`ai_image_classifier_hyperparameter_tuning.ipynb`**  
  Performs hyperparameter tuning to optimize the model's performance.

- **`ai_image_classifier_model.ipynb`**  
  Defines, trains, and evaluates a CNN model for image classification.

- **`GradioImageClassifier.ipynb`**  
  Deploys the trained model using a Gradio interface, allowing users to test images.

---
##  **Preprocessing** (`ai_image_classifier_data_cleanup.ipynb`)

### 1. Import Dependencies  
Ensure the following dependencies are installed:

```python
import os
import pandas as pd
from PIL import Image
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
```

### 2. Load Dataset
- The dataset is loaded from train.csv, containing file names and labels.

### 3. Resize Images
- All images are resized to 64x64 pixels using the Image.LANCZOS filter and stored in compressed_data/.

### 4. Convert Images to Numpy Arrays
- Images are converted into floating-point NumPy arrays.

### 5. Handle Grayscale Images
- Grayscale images are converted to RGB format for consistency.

### 6. Normalize Pixel Values
- Pixel values are scaled between 0 and 1 by dividing by 255.

### 7. Prepare Data for Training
- Extracts labels (y) from the dataset.
- Converts images and labels into NumPy arrays.

### 8. Train-Test Split
- The dataset is split into 80% training / 20% testing using a fixed random seed.

### 9. Data Augmentation
- To improve model performance, data augmentation is applied:
   - Random rotation (20 degrees)
   - Random translation (horizontal & vertical shift)
   - Random zoom
   - Random horizontal flip
   - For each original image, five augmented images are generated.

### 10. Save Preprocessed Data
- The preprocessed dataset is saved as ai_image_classifier_small_img.pkl, containing:
   - X_train: Augmented training imagesX_test: Test image
   - y_train: Augmented training labels
   - y_test: Test labels

### 11. Load Preprocessed Data
- The dataset can be reloaded later using pickle.

---
## **Model Implementation** (`ai_image_classifier_model.ipynb`)
### 1. Load Preprocessed Data
Loads data from ai_image_classifier_small_img.pkl.

### 2. Train-Test Split
- Further splits training data into 80% training / 20% validation.

### 3. Define the CNN Model
- A Convolutional Neural Network (CNN) is implemented using TensorFlow/Keras:
  - Conv2D layers with ReLU activation
  - MaxPooling layers for dimensionality reduction
  - Fully connected Dense layers
  - Output layer with softmax activation (binary classification)
  
### 4. Compile and Train the Model
- Optimizer: Adam
   - Loss function: Sparse categorical crossentropy
   - Epochs: 10
   - Validation data used during training
  
### 5. Evaluate the Model
- The trained model is evaluated using the test dataset to measure accuracy and loss.

---
## **Gradio Interface (`GradioImageClassifier.ipynb`)

### 1. Load the Trained Model
- The trained model is downloaded and loaded using TensorFlow.

### 2. Define the Image Classification Function
- Resizes input images to match the model’s input shape.
- Normalizes pixel values.
- Classifies whether an image is AI-generated or real.

### 3. Deploy with Gradio
- A Gradio UI is built with:
   - Image upload component
   - Label output displaying classification results
   - Simple UI with title and description
---
## **Results
### Predictive Analysis and Classification Threshold

Applying our trained model to classify images, we optimized the decision-making process by setting a standard rounding function (`> 0.5`).
- **Threshold Decision:**
  - If **P(AI-Generated) > 0.5**, the image is flagged as **AI-generated**.
  - If **P(AI-Generated) ≤ 0.5**, the image is classified as **real**.




### **License**
[Apache License Version 2.0](https://www.apache.org/licenses/LICENSE-2.0)
