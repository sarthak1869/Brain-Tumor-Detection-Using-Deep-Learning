
Brain-Tumor-Detection-Using-Deep-Learning
Deep learning-based brain tumor classification using LeNet, ResNet-50, and VGG-16 on MRI scans.

Brain Tumor Detection using LeNet, ResNet-50, and VGG-16

This project focuses on the detection of brain tumors using deep learning techniques. We implement and compare three popular convolutional neural network architectures: LeNet, ResNet-50, and VGG-16. The models are trained to classify MRI images of brain tumors into four categories:

- Glioma
- Meningioma
- Pituitary
- No Tumor

 # Objective

To build a robust deep learning pipeline that can accurately detect and classify brain tumors into one of the four defined classes using CNN-based models.

# Dataset

The dataset used consists of brain MRI images labeled into four categories. Each image was preprocessed and resized to match the input requirements of the respective models.

The dataset used for this project can be downloaded from the following link:

🔗 [Download Brain Tumor Dataset (Kaggle)](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)

- Image Size:
  - LeNet: 32x32
  - VGG-16 and ResNet-50: 224x224
- Classes: Glioma, Meningioma, Pituitary, No Tumor

# Models Used

- LeNet: A simple CNN model for baseline comparisons.
- ResNet-50: A deep residual network with 50 layers to learn more complex patterns.
- VGG-16: A 16-layer CNN known for its depth and simplicity.

# Model Evaluation

Each model was trained and evaluated on a train-validation-test split. Evaluation metrics include:

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix
- ROC AUC Score

# Results

| Model     | Accuracy |
|-----------|----------|
| LeNet     | 70%      |
| ResNet-50 | 96%      |
| VGG-16    | 93%      |

> Replace `XX%` with your actual results after evaluation.

# Requirements

Install dependencies using pip:

```bash
pip install -r requirements.txt
```

#  How to Run

1. git clone https://github.com/yourusername/brain-tumor-detection.git
cd brain-tumor-detection

2. Download the dataset and paste the training and testing in the same folder where your python file is stored
3. Run Each file a .h5 file will be stored in your folder through which you can implement gradio without everytime training the model once trained the model file will be stored.


# Acknowledgements
MRI images sourced from open datasets.

TensorFlow/Keras and PyTorch used for model development.

# Contact
For any queries or suggestions, feel free to reach out:

Sarthak Teli
[sarthak.teli18@gmail.com]


