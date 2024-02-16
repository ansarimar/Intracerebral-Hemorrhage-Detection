# -*- coding: utf-8 -*-
"""

@author: mansari
"""

import numpy as np
import keras
from keras.models import Model
from keras.layers import Dense,concatenate, Activation, MaxPooling2D, MaxPooling3D
from keras.layers import Conv2D, add, Input,Conv2DTranspose, Conv3D, GlobalAveragePooling2D
from keras.optimizers import SGD,Adam
from keras import losses
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot as plt
import math
import h5py
from keras.initializers import RandomNormal
#from preprocess_CT_image import load_scan, get_pixels_hu, write_dicom, map_0_1,windowing2
from keras.layers import BatchNormalization as BN
from keras import backend as K
def read_hdf5(file):
    with h5py.File(file, 'r') as hf:
        data = np.array(hf.get('data'))
        labels = np.array(hf.get('label'))
   
        return data,labels
#
batch_size = 32
data_train,_ = read_hdf5('D:\\Maryam-Dataset\H5-files\predict_dataset.h5') # data_train shape is (batch,num_ slices,512,512)

with h5py.File('D:\Maryam-Dataset\H5-files\predict_bbox.h5','r') as hf:
    bbox_arrays = np.array(hf.get('bbox'))

data_train = np.reshape(data_train,(-1,512,512,1)) # all ct slices 
data_train [data_train>80] = 80
data_train[data_train<0] = 0
data_train = data_train/80
labels_train = (np.reshape(bbox_arrays,(-1,5))[:,0]).astype('int8') # if ICH exist in the slice the value is 1

datagen = ImageDataGenerator(rotation_range=20, zoom_range = 0.25,
                             width_shift_range=0.2, height_shift_range=0.2,
                             horizontal_flip=True, validation_split=0.2,
                             data_format = 'channels_last')


base_model = keras.applications.xception.Xception(
        include_top=False, weights='imagenet', input_shape= None)

inputs = Input(shape=(512,512,1),name='input')
conv1 = MaxPooling2D()(inputs)
conv1 = Conv2D(3, (3,3), strides=(1,1), activation='relu', padding='valid')(conv1)


base_out = base_model(conv1)

x = GlobalAveragePooling2D()(base_out)
#  fully-connected layer
x = Dense(1024, activation='relu')(x)
# and a logistic layer 
output = Dense(1, activation='sigmoid')(x)


model = Model(inputs=[inputs], outputs=[output])

model.summary()

for layer in base_model.layers:
    layer.trainable = False
    

model.compile(optimizer='rmsprop',loss=losses.binary_crossentropy, metrics=['acc'])

#generates batches of augmented data 
model.fit_generator(datagen.flow(data_train, labels_train, batch_size=batch_size),
                    steps_per_epoch=10, epochs=30)
##



    
for layer in base_model.layers:
    layer.trainable = True
        


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

model.fit_generator(datagen.flow(data_train, labels_train, batch_size=batch_size),
                    steps_per_epoch=10, epochs=30, callbacks=[lrate],verbose=2)
base_model.save_weights('D:\\Maryam-Dataset\H5-files\Weights/base_model_0_80.h5')
#
#model.load_weights('D:\\Maryam-Dataset\H5-files\Weights\ICH_exist_inslice.h5')z
#model.save('D:\\Maryam-Dataset\H5-files\Weights\base_model.h5')
#data_test,_ = read_hdf5('D:\\Maryam-Dataset\H5-files\spotlight_dataset.h5') # data_train shape is (batch,512,512,num_ slices)
#data_test [data_test>240] = 240
#data_test[data_test<-240] = -240
#data_test = data_test/240
#
#shape = data_test.shape
#labels_predicted = np.zeros((shape[0],shape[1],1))
#
#for i in range (shape[0]):
#    labels_predicted[i] = model.predict(data_test[i,:,:,:,None],batch_size=32,verbose=1)
 









# This part find and saves m slices for each patient in data_testwith 
# highest probability of ICH obtained from the above network
data_test,labels_test = read_hdf5('D:\\Maryam-Dataset\H5-files\spotlight_dataset.h5') # data_train shape is (batch,512,512,num_ slices)
def m_ich_slices(data_test,labels_predicted,m=5):
    
    shape = data_test.shape
    data_test_m = np.zeros((shape[0],m,512,512),dtype='int')
    for i in range (shape[0]):
    
        index = np.sort(np.argpartition(labels_predicted[i,:,0], -m)[-m:])#find m slices with highest ICH probability and sort them
        # if indeces are not in order find m consecuitive slices 
        if index[m-1]-index[0] != m-1:
            k = np.median(index).astype('int')
            index = [k-2,k-1,k,k+1,k+2] #!!! This line should be changed to match any number it only works for m=5'''
            print(i)
            print(index)
                
        data_test_m[i] = data_test[i,index]
        
    return data_test_m
    
data_test_m = m_ich_slices(data_test,labels_predicted)

#write_hdf5(data_test_m,labels_test, 'D:\\Maryam-Dataset\H5-files\spotlght_dataset_5slice.h5')
