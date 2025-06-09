Onco-Vision: Deep Learning for Tumor Cell Detection in Breast Cancer

This project presents a deep learning-based solution for classifying breast cancer histopathological images as benign or malignant using DenseNet-201, ResNet-101, and InceptionV3 architectures. Our approach leverages transfer learning and image augmentation techniques to improve classification accuracy, with DenseNet-201 achieving up to 93% accuracy on the BreakHis 400x dataset.

Table of Contents

Overview

Objectives

Dataset

Model Architecture

Usage

Results

Comparison

Conclusion

Future Work

Team

References

Overview
Breast cancer remains one of the most prevalent life-threatening diseases. Histopathological analysis is a gold standard for diagnosis but is time-consuming and subjective. Our project, Onco-Vision, addresses this gap using deep convolutional neural networks (CNNs)—particularly DenseNet-201—to automate and enhance accuracy in tumor cell classification.

Objectives

Classify histopathological images into benign or malignant.Leverage transfer learning with pre-trained CNN architectures.Compare DenseNet-201 performance with ResNet-101 and InceptionV3.Implement robust preprocessing and augmentation techniques.Evaluate performance using metrics like accuracy, ROC-AUC, and confusion matrix.

Dataset

Dataset Name: BreakHis 400x

Image Type: Histopathology images (benign & malignant)

Resolution: 224x224

Train/Validation Split: 80% / 20%

Batch Size: 16

Magnification: 400x

Model Architecture

Primary Model: DenseNet-201 (Pretrained on ImageNet)

Other Models: ResNet-101, InceptionV3

Framework: TensorFlow 2.x with Keras

Platform: Google Colab

Preprocessing: Normalization, augmentation (flipping, rotation)

Activation Functions: ReLU, Softmax

Evaluation Metrics: Accuracy, Precision, Recall, F1-Score, ROC-AUC

Usage

Use the provided notebook to:

    Load and preprocess the dataset
    
    Train the DenseNet201 model
    
    Visualize training and validation performance
    
    Evaluate the model with ROC and confusion matrix

Results

Model	Accuracy	AUC Score

DenseNet-201	93%	0.88

InceptionV3	87%	0.75

ResNet-101	70%	0.56

DenseNet201 achieved the highest accuracy and stability.

ROC curves and confusion matrices showed strong separability between classes.

Comparison
DenseNet201: Efficient and high accuracy; best generalization.

InceptionV3: Lightweight and fast but slightly less accurate.

ResNet-101: Prone to overfitting and less stable.

Conclusion

DenseNet201 proved to be the most suitable model for classifying histopathology images, outperforming others in both accuracy and generalizability. This model can assist pathologists by providing a second opinion and speeding up diagnosis.

Future Work

Incorporate Explainable AI (XAI) techniques (e.g., Grad-CAM).

Expand to multi-modal datasets (e.g., clinical + imaging).

Deploy to edge/mobile devices for real-time diagnostics.

Use larger, more diverse datasets for better generalization.

Team
Jhanvi Acharya (ENG21CS0170)

K Vidyashree (ENG21CS0175)

Kamlesh N (ENG21CS0179)

Kavitha N (ENG21CS0185)

Supervised by Dr. T. Poongodi, Department of CSE, Dayananda Sagar University.

References
BreakHis Dataset

TensorFlow & Keras Documentation

Research papers listed in the Literature Survey (see project report)
