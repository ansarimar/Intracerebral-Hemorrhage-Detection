# -*- coding: utf-8 -*-
"""

@author: mansari
"""

import numpy as np
import keras
from keras.models import Model
from keras.layers import Dense,concatenate, Activation, MaxPooling2D, MaxPooling3D
from keras.layers import Conv2D, add, Input,Conv2DTranspose, Conv3D, GlobalAveragePooling2D
from keras.layers import TimeDistributed, ConvLSTM2D
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
def read_hdf5(file):
    with h5py.File(file, 'r') as hf:
        data = np.array(hf.get('data'))
        labels = np.array(hf.get('label'))
   
        return data,labels
#
batch_size = 32
data_train,labels_train = read_hdf5('D:\\Maryam-Dataset\H5-files\predict_dataset.h5') # data_train shape is (batch,512,512,num_ slices)

data_train_m = m_biggest_bbox(data_train,m=5).astype('int32')

data_train_m[data_train_m>80] = 80
data_train_m[data_train_m<0] = 0
data_train_m = data_train_m[:,:,:,:,None]/80
#labels_train = labels_train[:,None,None,None,None]



#datagen = ImageDataGenerator(rotation_range=20, zoom_range = 0.25,
#                             width_shift_range=0.2, height_shift_range=0.2,
#                             horizontal_flip=True, validation_split=0.2,
#                             data_format = 'channels_last')
#
base_model = keras.applications.xception.Xception(
        include_top=False, weights=None, input_shape= None)



base_model.load_weights('D:\\Maryam-Dataset\H5-files\Weights/base_model_0_80.h5')

intermediate_layer_model = Model(inputs=base_model.input,
                                 outputs=base_model.get_layer
                                 ('block13_sepconv1_act').input)

inputs = Input(shape=(5,512,512,1),name='input')

conv0 = TimeDistributed(MaxPooling2D())(inputs)
conv0 = TimeDistributed(Conv2D(3, (3,3), strides=(1,1), activation='relu', 
                               padding='valid'))(conv0)
conv0 = TimeDistributed(Conv2D(3, (3,3), strides=(2,2), activation='relu', 
                               padding='valid'))(conv0)

base_out = TimeDistributed(intermediate_layer_model)(conv0)

conv1 = TimeDistributed(Conv2D(3, (3,3), strides=(1,1), activation='relu', 
                               padding='valid'))(base_out)

conv1 = ConvLSTM2D(64, (3,3), strides=(1,1), return_sequences= True)(conv1)
conv2 = ConvLSTM2D(64, (3,3), strides=(1,1))(conv1)

x = GlobalAveragePooling2D()(conv2)
# let's add a fully-connected layer
x = Dense(1024, activation='relu')(x)
# and a logistic layer -- let's say we have 200 classes
outputs = Dense(1, activation='sigmoid')(x)


model = Model(inputs=[inputs], outputs=[outputs])

model.summary()

for layer in intermediate_layer_model.layers:
    layer.trainable = False
    
#        
#


ADAM=Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
model.compile(optimizer=ADAM,loss=losses.binary_crossentropy, metrics=['acc'])

# generates batches of augmented data and 
#model.fit_generator(datagen.flow(data_train_m, labels_train, batch_size=32),
#                    steps_per_epoch=10, epochs=10)
##

inputs = [data_train_m]
hist_adam = model.fit(x=inputs,y=labels_train,batch_size=batch_size,epochs=20
                     ,validation_split=0.05, verbose=2, shuffle=True)



for layer in intermediate_layer_model.layers[66:]:
    layer.trainable = True
#        
#


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


hist_adam = model.fit(x=inputs,y=labels_train,batch_size=batch_size,epochs=50
                     ,validation_split=0.0, verbose=1, shuffle=True)


