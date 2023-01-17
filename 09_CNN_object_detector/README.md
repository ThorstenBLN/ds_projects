### Object detector CNN

#### target:
Build an Convolutional Neural Network that recognizes objects that I hold into the webcam

#### data: 
images of objects of 3 classes (bottle, pens, sunglasses) which I took with the webcam of my notebook and augmented with tensorflow.

#### files:
- CNN_AUG.ipynb: 
    - loads all the images 
    - augmentation of the images of the 3 classes (randomly flipped and rotated)
    - creation and fit of a CNN with 2 convolutional layers, 2 pooling layers and 1 dense layers
    - plots all test images and predicts their class
    - uses a tensorboard callback to analyse the training process

- Pretrained_model.ipynb:
    - 1. use the VGG16-model to predict the images without pretraining (not very precise)
    - 2. use only the convolutional part of the pretrained model. Replace the Dense Layers with 
         a new Dense Layer, train it with the train images and predict the test images (accuracy: 100%)