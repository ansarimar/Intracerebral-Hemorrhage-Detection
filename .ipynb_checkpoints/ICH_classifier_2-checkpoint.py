# -*- coding: utf-8 -*-
"""

@author: mansari
"""

import numpy as np
from keras.models import Model
from keras.layers import Dense,concatenate, Activation, MaxPooling2D, MaxPooling3D
from keras.layers import Conv2D, add, Input,Conv2DTranspose, Conv3D
from keras.optimizers import SGD,Adam
from keras import losses
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot as plt
import math
import h5py
from keras.initializers import RandomNormal
#from preprocess_CT_image import load_scan, get_pixels_hu, write_dicom, map_0_1,windowing2
from keras.layers import BatchNormalization as BN
from biggest_BBOX import m_biggest_bbox
from keras import backend as K
import keras

def read_hdf5(file):
    with h5py.File(file, 'r') as hf:
        data = np.array(hf.get('data'))
        labels = np.array(hf.get('label'))
   
        return data,labels



m = 5
batch_size = 32
data_train,labels_train = read_hdf5('D:\\Maryam-Dataset\H5-files\predict_dataset.h5') # data_train shape is (batch,512,512,num_ slices)

data_train_5 = m_biggest_bbox(data_train)
data_train_5 [data_train_5>240] = 240
data_train_5[data_train_5<-240] = -240
data_train_5 = np.moveaxis(data_train_5,1,3)/240

labels_train = labels_train[:,None,None,None]

datagen = ImageDataGenerator(rotation_range=0, zoom_range = 0.0,
                             featurewise_center=True,featurewise_std_normalization=True,
                             width_shift_range=0.1, height_shift_range=0.1,
                             horizontal_flip=True, validation_split=0.2,
                             data_format = 'channels_last')
datagen.fit(data_train_5)


inputs = Input(shape=(512,512,5),name='input')

conv1 = Conv2D(32, (3,3), strides=(2,2), activation='relu', padding='valid',
               name='conv0')(inputs)
conv1 = BN()(conv1)
conv1 = Activation('relu')(conv1)

conv1=Conv2D(48, (3,3),strides=(2,2), padding='valid',
             name='conv1')(conv1)
conv1 = BN()(conv1)
conv1 = Activation('relu')(conv1)

conv2 = Conv2D(96, (3,3),strides=(2,2), padding='same',dilation_rate=(1,1),
               name='conv2')(conv1)
conv2 = BN()(conv2)
conv2 = Activation('relu')(conv2)

conv3 = Conv2D(128, (3,3),strides=(2,2), padding='same',dilation_rate=(1,1),
               name='conv3')(conv2)
conv3 = BN()(conv3)
conv3 = Activation('relu')(conv3)

conv4 = Conv2D(256, (3,3),strides=(2,2), padding='same',dilation_rate=(1,1),
               name='conv4')(conv3)
conv4 = BN()(conv4)
conv4 = Activation('relu')(conv4)

conv5 = Conv2D(32, (3,3),strides=(2,2), padding='same',dilation_rate=(1,1),
               name='conv5')(conv4)
conv5 = BN()(conv5)
conv5 = Activation('relu')(conv5)
conv6 = Conv2D(32, (3,3),strides=(2,2), padding='same',dilation_rate=(1,1),
               name='conv6')(conv5)
conv6 = BN()(conv6)
conv6 = Activation('sigmoid')(conv6)

conv7 = Conv2D(1,(4,4))(conv6)
model = Model(inputs=[inputs], outputs=[conv7])

model.summary()


#ADAM=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
#sgd=SGD(lr=0.01, momentum=0.9, decay=0.01, nesterov=True)
#model.compile(optimizer=sgd,loss=losses.binary_crossentropy,metrics=['acc'])

def step_decay(epoch):
   initial_lrate = 0.1
   drop = 0.5
   epochs_drop = 10.0
   lrate = initial_lrate * math.pow(drop,  
           math.floor((1+epoch)/epochs_drop))
   return lrate
lrate = keras.callbacks.LearningRateScheduler(step_decay)

model.compile(optimizer=SGD(lr=0.0001, momentum=0.9,decay=0.1,nesterov=True),
              loss=losses.binary_crossentropy,metrics=['acc'])

ADAM=Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
model.compile(optimizer=ADAM,loss=losses.binary_crossentropy,metrics=['acc'])

## generates batches of augmented data and 
#model.fit_generator(datagen.flow(data_train_5, labels_train, batch_size = batch_size),
#                    steps_per_epoch=20, epochs=50,verbose=1)

hist_adam = model.fit(x=data_train_5,y=labels_train,batch_size=batch_size,epochs=20
                     ,validation_split=0.2, verbose=1, shuffle=True)



