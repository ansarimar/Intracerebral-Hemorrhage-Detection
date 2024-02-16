"""


@author: mansari
"""

import numpy as np
from keras.models import Model
from keras.layers import Dense,concatenate, Activation, MaxPooling2D, MaxPooling3D
from keras.layers import Conv2D, add, Input,Conv2DTranspose, Conv3D, Reshape
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
m = 5
batch_size = 32
data_train,labels_train = read_hdf5('D:\\Maryam-Dataset\H5-files\predict_dataset.h5') # data_train shape is (batch,512,512,num_ slices)

data_train_5 = m_biggest_bbox(data_train)
data_train_5 [data_train_5>240] = 240
data_train_5[data_train_5<-240] = -240
data_train_5 = np.moveaxis(data_train_5,1,3)/240

labels_train = labels_train[:,None,None,None]


inputs = Input(shape=(512,512,5),name='input')
reshape = Reshape((5,512,512,1))(inputs)

conv0 = Conv3D(16, (1,3,3), strides=(1,2,2), activation='relu', padding='valid',
               name='conv0')(reshape)

conv1=Conv3D(16, (1,3,3),strides=(1,1,1), activation='relu', padding='valid',
             name='conv1')(conv0)
conv1 = BN()(conv1)
conv1 = Activation('relu')(conv1)

conv2 = Conv3D(16, (1,3,3),strides=(1,1,1), padding='valid',dilation_rate=(1,1,1),
               name='conv2')(conv1)
conv2 = BN()(conv2)
conv2 = Activation('relu')(conv2)

conv3 = Conv3D(32, (1,3,3),strides=(1,2,2), padding='valid',dilation_rate=(1,1,1),
               name='conv3')(conv2)
conv3 = BN()(conv3)
conv3 = Activation('relu')(conv3)

conv3 = Conv3D(32, (1,3,3),strides=(1,2,2), padding='valid',dilation_rate=(1,1,1),
               name='conv32')(conv3)
conv3 = BN()(conv3)
conv3 = Activation('relu')(conv3)

conv3 = Conv3D(32, (1,3,3),strides=(1,2,2), padding='valid',dilation_rate=(1,1,1),
               name='conv33')(conv3)
conv3 = BN()(conv3)
conv3 = Activation('relu')(conv3)


conv4 = Conv3D(32, (1,3,3),strides=(1,1,1), padding='valid',dilation_rate=(1,1,1),
               name='conv4')(conv3)
conv4 = BN()(conv4)
conv4 = Activation('relu')(conv4)

conv5 = Conv3D(32, (3,3,3),strides=(2,2,2), padding='valid',dilation_rate=(1,1,1),
               name='conv5')(conv4)
conv5 = BN()(conv5)
conv5 = Activation('relu')(conv5)

conv6 = Conv3D(32, (2,3,3),strides=(2,2,2), padding='valid',dilation_rate=(1,1,1),
               name='conv6')(conv5)
conv6 = BN()(conv6)
conv6 = Activation('relu')(conv6)

conv7 = Conv3D(100,(1,3,3))(conv6)
conv7 = BN()(conv7)
conv7 = Activation('relu')(conv7)

conv7 = Conv3D(1,(1,4,4))(conv7)
conv7 = BN()(conv7)
conv7 = Activation('relu')(conv7)

conv7 = Reshape((-1,1,1))(conv7)
model = Model(inputs=[inputs], outputs=[conv7])

model.summary()


ADAM=Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
model.compile(optimizer=ADAM,loss=losses.binary_crossentropy, metrics = ['accuracy'])

#hist_adam = model.fit(x=data_train_5,y=labels_train,batch_size=batch_size,epochs=50
#                     ,validation_split=0.2, verbose=1, shuffle=True)





datagen = ImageDataGenerator(rotation_range=0, zoom_range = 0.0,
                             width_shift_range=0.1, height_shift_range=0.1,
                             horizontal_flip=True, validation_split=0.2,
                             data_format = 'channels_last')
#
# generates batches of augmented data 
train_generator = datagen.flow(data_train_5, labels_train, batch_size=32)
model.fit_generator(train_generator,steps_per_epoch=50, epochs=50, verbose=2)
##

