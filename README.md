# emotion_recognition

## Overview:
We performed the Emotion Recognition Task using Random Forest Classifier, Deep Neural Networks Model, Convolutional Neural Networks (CNN), Residual Network Model (ResNet), Convolutional Recurrent Neural Network Model (CRNN), and Transformer Model. One of the main challenges we faced is the lack of standard datasets and benchmarks along with the huge size of the dataset which made it hard for us to process it and execute the algorithms. In terms of recommendations, we would highly recommend carefully preprocess the data and select appropriate features that capture the relevant emotional cues. We also find that it is important to tune in appropriate parameters/hyperparameters so as to obtain optimum performance.

## Method:
The given dataset primarily consists of two sub-datasets namely training and testing. The training dataset consists of 3 features namely “id”, “emotion” and “pixels” wherein “id” represents the index number, “emotion” consists of all the types of emotions namely Angry, Disgust, Fear, Happy, Sad, Surprise and Neutral and “pixels” representing the data pertaining to the respective emotions. The testing dataset consists of primarily 2 features namely “id” and “pixels” which are used as an input to predict the “emotion” of the input variables. Given this data, we learnt that it is essential to carefully preprocess the data so as to obtain optimum efficiency.

## Pre-processing
For pre-processing, we firstly check if there are any missing/null values in the dataset post which we prepared the pixel data from training and testing data frames to be used as an input for the ML models. Secondly, for train, test and validation, we have only split the training dataset into training and validation sets.

Additionally, for pre-processing the data, we have split the “pixels” feature and reshaped the input images in the size 48x48 and we have also reshaped this feature based on every model’s requirements so as to better train the models and obtain the desired output. We have also used One-hot Encoding in some of the models as it was necessary for the model to improve its performance.

## Data augmentation
Moving on, we have implemented data augmentation for our preprocessing which we executed by setting up an Image Data Generator object with various data augmentation parameters which we used to generate augmented images for model training.

## Models
### Standard ML Baseline: Random Forest Classifier
Simple Machine Learning Baseline Model: We have used Random Forest Classifier for this task. Random Forest Classifier is a popular machine learning algorithm that is well-suited for classification tasks. We have chosen it mainly due to its ability to handle complex relationships, robustness to noise and outliers, feature importance estimation, and ensemble learning capabilities.

### Deep NN model and CNN
We have used Convolutional Neural Network as our Deep Neural Network Model. It is a type of neural network commonly used for image and video classification tasks. We chose it mainly due to its ability to handle complex relationships between input features and output labels, and their high accuracy in a wide range of image and speech recognition tasks.

### Residual Network (ResNet):
ResNet is a deep neural network which uses skip connections that allow the network to learn residual functions instead of directly learning the mapping between input and output. We have mainly considered to use it due to its ability to show excellence in image recognition tasks. One of its cardinal strength is to effectively train deep neural networks with hundreds of layers by addressing the problem of vanishing gradients that can occur during training.

Results:
According to our results, we can conclude that Convolutional Recurrent Neural Networks Model (CRNN) is the best performing model having an accuracy of 0.69. We achieved such a high performance primarily because of Data Augmentation. We have used data augmentation in this model which helps in increasing the diversity of training data, improving the generalization of the model as well as in producing efficient and robust models. We have computed the mean and standard deviations for all the models used which include Random Forest Classifier Model, Deep Neural Networks Model (DNN), Convolutional Neural Networks Model (CNN), Transformer Model, Convolutional Recurrent Neural Networks Model (CRNN) and Residual Networks Model (ResNet). The average mean and standard deviation for our results are 0.18and 0.09 respectively and the mean and standard deviation of our best model (i.e CRNN Model) are 0.16 and 0.23 respectively. Secondly, the Convolutional Neural Networks Model (CNN) and the Transformer model are the worst performing models as per the above table. This can be seen primarily due to not using data augmentation for each of these models. It can also be seen that the epoch input also plays a cardinal role in the success of the models. In this case, for CNN and Transformer, we have kept a significantly low epoch value for the purpose of training the model which thereby implies the low accuracy score.

We tried different epoch values for our best model so as to check for the optimum result however, due to technical limitations we were unable to attain the desired value for the epochs used. We have set epochs as 200 for the CRNN model, which is our best model, and was the highest which could work using the available resources we had at our disposal.

## Summary:
We would recommend using the Convolutional Recurrent Neural Networks Model (CRNN) for this task as the model that performed the best was CRNN achieving an accuracy of 0.69. Due to CRNNs features such as capturing of temporal and spatial information, robustness to noise, flexibility and state-of-the-art performance makes CRNN the best model to use. Data Augmentation also plays a vital role in the success of this model which is primarily responsible for improving the performance and robustness of a CRNN model. It not only improves generalization performance but also increases data efficiency in training the CRNN model. Secondly, as stated above, epoch values also have a significant impact on the accuracy of a model. Having used an epochs value of 200 for the CRNN model, makes CRNN a better performer than the rest.
Post submitting the models on Kaggle, we received a score of 0.65, which can be viewed as a good performance especially in comparison with peer groups. The score has some room for improvement whatsoever and can be done further by increasing the epochs value. In our case we could not achieve it due to technical inability however, given a favourable technical infrastructure, we could attain a better overall score.
