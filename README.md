# Facial-Expression-Recognition-in-the-Wild-with-CNN-ResNet-GoogleNet-Inception-and-VGG-Models
![Project Logo](https://github.com/SBJ2000/Facial-Expression-Recognition-in-the-Wild-with-CNN-ResNet-GoogleNet-Inception-and-VGG-Models/blob/main/Images/Logo.jpg)
## Introduction :
Facial Expression Recognition (FER) plays a crucial role in understanding human emotions, with applications ranging from human-computer interaction to affective computing. This project aims to develop a robust image classification model to recognize seven emotions (joy, surprise, fear, anger, disgust, neutral, and sadness) using the FER 2013 dataset, which captures expressions in natural environments ("in-the-wild").
## Models Used :
To achieve the project's goal, various deep learning models were tested to compare performance, including:

* Convolutional Neural Network (CNN)
* ResNet
* GoogleNet
* Inception
* VGG
## Working Environment :

[![Google Colab](https://img.shields.io/badge/Google%20Colab-Primary%20Platform-FFD700)](https://colab.research.google.com/)
    
    Primary platform for developing and training the facial expression recognition models, leveraging cloud computing and free GPU resources.
[![Python](https://img.shields.io/badge/Python-Central%20to%20Development-3776AB)](https://www.python.org/)

    Central to model development and backend implementation, utilizing libraries such as TensorFlow and PyTorch.

[![Visual Studio](https://img.shields.io/badge/Visual%20Studio-Backend%20%26%20Frontend%20Development-5C2D91)](https://visualstudio.microsoft.com/)

    Used for both backend and frontend development, offering robust debugging tools and integrated source control.
[![React JS](https://img.shields.io/badge/React%20JS-Frontend%20Development-61DAFB)](https://reactjs.org/)
    
    Employed for frontend development, providing a component-based architecture for building interactive web applications.

## Model Architectures and Training :
###Convolutional Neural Network (CNN)

#### Model Definition
The CNN model includes:

* Convolutional layers with ReLU activation
* MaxPooling layers to reduce spatial dimensions
* Dropout layers to mitigate overfitting
* Fully connected layers with ReLU activation
* Softmax activation for the output layer

#### Data Exploration and Preprocessing

The FER 2013 dataset consists of 7178 images. Images were converted to grayscale, normalized, and one-hot encoded for training.

#### Training and Evaluation
* Trained over 100 epochs with a batch size of 128.
* Achieved an accuracy of approximately 63.36% on the test set.

### ResNet Model
#### Model Definition
The ResNet34 architecture was used, pretrained on ImageNet, with the final fully connected layer adjusted for seven emotion classes.

#### Data Exploration and Preprocessing

Similar preprocessing steps as the CNN model, with additional data augmentation techniques like random horizontal flips and rotations.

#### Training and Evaluation
* Trained over 20 epochs using the Adam optimizer.
* Achieved a validation accuracy of approximately 64.97%.

### GoogleNet Model
#### Model Definition

Utilizes the pre-trained GoogleNet architecture, with the classification layer modified for seven emotion classes.

#### Data Exploration and Preprocessing
Employed similar preprocessing and data augmentation techniques as previous models.

#### Training and Evaluation
* Trained over 20 epochs using the Adam optimizer.
* Achieved a validation accuracy of approximately 63.61%.

### Inception Model
#### Model Definition

Built using the pre-trained Inception V3 architecture, with the final layer modified for seven emotion classes.

#### Data Exploration and Preprocessing

Similar preprocessing and data augmentation techniques were used.

#### Training and Evaluation
* Trained over 20 epochs using the Adam optimizer.
* Achieved the highest validation accuracy of approximately 68.65%.

### VGG Model
#### Model Definition

Constructed using the pre-trained VGG16 architecture, with the final fully connected layer adjusted for seven emotion classes.

#### Data Exploration and Preprocessing

Employed similar preprocessing steps as other models.

#### Training and Evaluation

* Trained over 20 epochs using the Adam optimizer.
* Achieved a validation accuracy of approximately 24.64%.

## Graphical User Interface

A user-friendly graphical interface was developed using React, coupled with a Python backend to enhance the interactive experience of utilizing the facial expression recognition model. The API facilitates seamless communication between the frontend and backend, ensuring a responsive and intuitive user interface for real-time facial expression analysis.

![GUI](https://github.com/SBJ2000/Facial-Expression-Recognition-in-the-Wild-with-CNN-ResNet-GoogleNet-Inception-and-VGG-Models/blob/main/Images/GUI.png)

## Additional Information

For additionnal information especially for the code you can read the report in the report folder.

## Conclusion
This data science mini-project focused on developing and evaluating facial expression recognition models for in-the-wild datasets using the FER 2013 dataset. Five different models were explored, each demonstrating unique architectures, training processes, and performance metrics. The Inception model achieved the highest validation accuracy, showcasing its architectural innovation and robust training process. Finally, a graphical user interface was developed to provide an interactive platform for utilizing the trained models, ensuring a user-friendly experience for facial expression analysis.