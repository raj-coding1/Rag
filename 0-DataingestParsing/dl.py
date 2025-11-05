#import all the required packages
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import models,layers
import matplotlib.pyplot as plt

#load the data sets

(x,y),(xt,yt) = tf.keras.datasets.mnist.load_data()
x = 1/255.
xt = 1/255.
#train the model

model = models.Sequential([
    tf.keras.Input(shape=(28, 28)),
    layers.Flatten(),
    layers.Dense(16,activation='relu'),
    layers.Dense(10,activation='softmax')
])
model.compile(optimizer='adam', loss = 'sparse_categorical_crossentropy', metrics= ['accuracy'])
history= model.fit(x,y,epochs = 30, validation_data=(xt, yt))
test_loss , test_accuracy = model.evaluate(xt,yt)
# print(f'loss is : {} and accuracy is : {}'.format(test_loss))
print(f'loss is : {test_loss} and accuracy is : {test_accuracy}')