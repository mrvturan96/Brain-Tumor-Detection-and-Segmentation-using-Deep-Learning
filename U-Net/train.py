import numpy as np
from model import unet_model
import matplotlib.pyplot as plt
import os
import logging

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

img_size = 120

num_epoch = 3

# Load the dataset as npy format.
print("Loading dataset.")
train_X = np.load('C:/Users/merid/Documents/DeepHealth/MRI/x_{}.npy'.format(img_size))
train_Y = np.load('C:/Users/merid/Documents/DeepHealth/MRI/y_{}.npy'.format(img_size))
print("Dataset loaded")

print("Loading the model")
model = unet_model()
# model.summary()
print("The model loaded")



# train the model
print("Starting training")
history = model.fit(train_X, train_Y, validation_split=0.25, batch_size=16, epochs= num_epoch, shuffle=True,  verbose=1,)

'''
# For binary metrics
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()'''

# Plot training & validation accuracy values

plt.plot(history.history['dice_coef'])
plt.plot(history.history['val_dice_coef'])
plt.title('model dice_coef')
plt.ylabel('dice_coef')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()




# save the model
model.save_weights('dice_weights_{}_{}.h5'.format(img_size,num_epoch))
