# Keras Model Basic Tutorial

**"keras_model_tutorial_01.py"** is an attempt to help Bigineers understand the complete flow of building a Keras Neural Network
it is a simple classification problem where divided into two classes.

**Our Objective :** We will train a small CNN network on it and find the confusion matrix.

The Image datset for "Keras_model_tutorial.py" can be downloaded from the following link
https://drive.google.com/file/d/18-XchvcQcmLMkNXNoiFruZfx340hnaUR/view?usp=sharing

**_Folder Structure_**
1. All images are present in just 1 folder, there is no seperate directory based on classes or Train and Test data.
2. Images are converted into ndarray and loaded into memory.

**Problems**
1. Data loading may fail for Low Memory Devices as all the images are loaded into memory before  fitting the model


**"keras_model_tutorial_02.py"** is an attempt to improve the pipeline for training.
In this pipeline we use _keras.fit_generator_ this helps us in a unique way as all the image data is not loaded into the memory before the training process.
.
