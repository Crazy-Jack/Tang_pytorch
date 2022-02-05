# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 11:32:37 2020

@author: pku
"""
#%%
import tensorflow
import os
import numpy as np
from PIL import Image
from tensorflow.keras import applications
from tensorflow.keras import optimizers
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.callbacks import ModelCheckpoint,EarlyStopping
from tensorflow.keras.layers import Flatten, Dense, Dropout

import matplotlib.pyplot as plt 
import h5py
#%%


nb_train_samples = 49000
nb_validation_samples = 1000



#%%

data = np.load('../Rsp.npy')
test = np.load('../valRsp.npy')


n= nb_train_samples
train_x = np.load('../train_x.npy')
val_x   = np.load('../val_x.npy')

#%%
from tensorflow.keras.layers import Conv2D,BatchNormalization, MaxPooling2D, DepthwiseConv2D, Activation, GaussianNoise, LocallyConnected2D
from tensorflow.keras import regularizers
from tensorflow.keras import activations
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K
import tensorflow as tf

#model = load_model('model.hdf5')

class L1():
    def __init__(self, l1=0.00001):
        self.l1 = l1
    
    def __call__(self, weight_matrix):
        


#from tensorflow.keras.models import load_model
#model=load_model('my_trained_model.h5')

        return self.l1 * K.mean(K.abs(weight_matrix))**2/K.mean(weight_matrix**2)
    
    def get_config(self):

        return {'l1': float(self.l1)}



def mean_squared_error_noise(y_true, y_pred):
    return K.mean(K.square(K.relu(K.abs(y_pred - y_true)-0.1)), axis=-1)

# to deal with Failed to get convolution algorithm. 
config = tf.compat.v1.ConfigProto()#####previous1: tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.InteractiveSession(config=config)

#alexnet = load_model('alexnet.h5')
model = Sequential()
model.add(Conv2D(30, (5, 5),strides=(1,1), input_shape=(50,50,1),kernel_regularizer=L1(0.001)))
model.add(MaxPooling2D((2, 2)))
model.add(BatchNormalization())
model.add(Activation(activations.sigmoid))
model.add(Dropout(0.3))
model.add(Conv2D(30, (5, 5),strides=(1,1),kernel_regularizer=L1(0.001)))
model.add(MaxPooling2D((2, 2)))
model.add(BatchNormalization())
model.add(Activation(activations.sigmoid))
model.add(Dropout(0.3))
model.add(Conv2D(30, (3, 3),strides=(1,1),kernel_regularizer=L1(0.001)))
model.add(BatchNormalization())
model.add(Activation(activations.sigmoid))
model.add(Dropout(0.3))
model.add(Conv2D(30, (3, 3),strides=(1,1),kernel_regularizer=L1(0.001)))
model.add(BatchNormalization())
model.add(Activation(activations.sigmoid))
model.add(Flatten())
model.add(Dense(1,kernel_regularizer=regularizers.l1(0.0001)))


#weights = alexnet.layers[1].get_weights()
#model.layers[0].set_weights(weights)
#model.layers[0].trainable = False

#adadelta=optimizers.Adadelta(lr=0.001, rho=0.95, epsilon=1e-06)
Adam = optimizers.Adam(lr=0.0002, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
model.compile(loss=mean_squared_error_noise, optimizer=Adam, metrics=['mse'])
model.summary()
model.save_weights('initial.hdf5')

#%%
R = np.zeros((1,299))
VE = np.zeros((1,299))
for i in range(10):
    neuron = i
    ROI = data[(nb_train_samples-n):nb_train_samples,neuron]
    
    train_y = np.reshape(ROI,(n,1))
    val_y   = np.reshape(test[:,neuron],(1000,1))
    
    model.load_weights('initial.hdf5')
    
    filepath="Cell %d" % (neuron) + ".hdf5"
    
    earlyStopping=EarlyStopping(monitor='val_mean_squared_error', patience=200, verbose=1, mode='auto')
    saveBestModel = ModelCheckpoint(filepath, monitor='val_mean_squared_error', verbose=1, save_best_only=True, mode='auto')
    
    
    callbacks_list = [earlyStopping,saveBestModel]
    
    model.fit(train_x, train_y, epochs=50, batch_size=20, validation_data=(val_x, val_y), callbacks=callbacks_list)
    
    # test trained model
    model.load_weights(filepath)
    pred= model.predict(train_x)
    pred1 = model.predict(val_x)
    u2=np.zeros((2,nb_validation_samples))
    u2[0,:]=np.reshape(pred1,(nb_validation_samples))
    u2[1,:]=np.reshape(val_y,(nb_validation_samples))
    
    
        
    c2=np.corrcoef(u2)
    R[0,neuron] = c2[0,1]
    print(R)
    VE[0,neuron] = 1-np.var(pred1-val_y)/np.var(val_y)
    print(VE)
    
    file_handle = open('Record1.txt',mode = 'w')
    file_handle.write(str(R))
    file_handle.write('\n')
    file_handle.write(str(VE))
    file_handle.close()
#%%
np.save('R.npy',R)
np.save('VE.npy',VE)