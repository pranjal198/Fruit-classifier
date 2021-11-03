# Fruit classifier

This is my first project with neural networks. It classifies 130 different fruits on the basis of a dataset with more than 90,000 images. 

I found the image dataset from [kaggle](https://www.kaggle.com/) by the name of [Fruits 360](https://www.kaggle.com/moltean/fruits). I split the dataset for training(80%) and validation(20%).then I added a preprocessing layer from TensorFlow.

I used the [ResNet](https://keras.io/api/applications/resnet/) model ResNet50 as my base model. then I added a global average pooling layer and a prediction layer. I trained the model with 10 epochs and achieved an accuracy of 99.84% on the trainig dataset and 99.90% on the validation dataset.