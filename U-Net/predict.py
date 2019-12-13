import os
import logging
from keras.models import Sequential
import matplotlib.pyplot as plt
import numpy as np
import random as r
from model import unet_model

def set_tf_loglevel(level):
    if level >= logging.FATAL:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    if level >= logging.ERROR:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    if level >= logging.WARNING:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
    else:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
    logging.getLogger('tensorflow').setLevel(level)

set_tf_loglevel(logging.FATAL)


# training examples with augmentation
total_examples = 8000

# img size of the mri images
img_size = 120

print("Loading dataset...")
train_X = np.load('C:/Users/merid/Documents/DeepHealth/MRI/x_{}.npy'.format(img_size))
train_Y = np.load('C:/Users/merid/Documents/DeepHealth/MRI/y_{}.npy'.format(img_size))
print("Dataset loaded.")

model = unet_model()

print("Loading the model...")
model.load_weights('weights/dice_weights_120_10.h5')
print("The model loaded.")


pred = model.predict(train_X[total_examples:total_examples+100]) # 8000 num with data aug

for n in range(6):
    i = int(r.random() * pred.shape[0])
    x = train_X[i+total_examples, 0, :, :]
    y = train_Y[i+total_examples, 0, :, :]
    y_predicted = pred[i, 0, :, :]
    combined_predicted = x + y_predicted

    fig = plt.figure(figsize=(15,10))


    plt.subplot(131)
    plt.title('Input:')
    plt.imshow(x,cmap='gray')

    plt.subplot(132)
    plt.title('Ground Truth')
    plt.imshow(x + y,cmap='gray')


    plt.subplot(133)
    plt.title('Prediction') 
    ##plt.imshow(combined_predicted,cmap='gist_heat')
    plt.imshow(combined_predicted,cmap='cividis')


    plt.show()
