# MNIST classification
## Project
MNIST is a labeled dataset of written digits (0-9).\
The purpose of this project is to train a neural network to be able to classify handwritten digits and it serves as an entry point for my personal machine learning journey.\
The notebooks directory contains Jupyter notebooks.
## Model
Firstly, I will use a deep neural network (dnn) to implement the classifier.
Later, I will come back and add a convolutional neural network (cnn) as a second model.
I will then compare the performance of the two models in this project and reflect upon the possible cause.
## Project structure and class diagrams
The project is composed of 3 modules:
1. The main module, which combines the different modules of the api in order to firstly train the model and then accept user input which can be classified.
2. The model module containing all models, the loss functions and optimization methods.
3. The data module constituted of utility functions for io, data preparation and manipulation as well as training, validation and evaluation of models based on their output.
