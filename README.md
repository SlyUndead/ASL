# American Sign Language Detection

This project aims to develop a system for detecting American Sign Language (ASL) letters using computer vision and deep learning techniques. The system employs a Convolutional Neural Network (CNN) model trained on a custom dataset to recognize and classify hand gestures representing letters in the ASL alphabet.


# Project Overview
The American Sign Language Detection system consists of two main components:
Hand Detection: The system uses the HandDetector module from the cvzone library to detect and track the user's hand in real-time video input from a webcam.
Sign Language Classification: The detected hand region is preprocessed and fed into a custom-trained CNN model for classification. The model predicts the corresponding ASL letter based on the hand gesture.


# Dataset

The project uses a custom dataset collected by the developers. The dataset consists of images of hand gestures representing the 26 letters of the ASL alphabet. The dataset is located in the Data folder and is organized into sub-folders for each letter.
Dependencies
The following Python libraries are required to run the project:

OpenCV
NumPy
cvzone
Keras (TensorFlow backend)


# Usage

Clone the repository to your local machine.
Install the required dependencies.
Ensure that the keras_model.h5 (pre-trained CNN model) and labels.txt (class labels) files are present in the Model directory.
Run the hand.py script.
The system will start capturing video from your webcam and detect and classify ASL letters in real-time.


# Model Training

The CNN model used in this project was trained on the custom dataset. If you wish to retrain the model or train a new model, you will need to provide your own dataset and implement the necessary training scripts. You can run the dataCollection.py script in order to collect pictures for your model to train on. Once the script has run press 's' on your keyboard to save the image. You will have to change the directory in the code based on the letter you are taking the image of. Next run the training.py code to train the model with the dataset created.


# Acknowledgements

The HandDetector and Classifier modules are from the cvzone library.
The Keras library and TensorFlow backend were used for building and training the CNN model.
