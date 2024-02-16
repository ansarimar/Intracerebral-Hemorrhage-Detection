# -*- coding: utf-8 -*-
"""
Created on Fri Mar  2 11:17:32 2018

@author:Maryam
Based on: https://www.kaggle.com/gzuidhof/full-preprocessing-tutorial
"""

import numpy as np # linear algebra
import pydicom
import os
import scipy.ndimage
import matplotlib.pyplot as plt






# Load the scans in given folder path
def load_scan(path):
    slices = [pydicom.read_file(path + '/' + s) for s in os.listdir(path)]
    slices.sort(key = lambda x: float(x.ImagePositionPatient[2]))
    try:
        #distance between slices, finds slice tkickness if not availabe
        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
    except:
        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)
        
    for s in slices:
        s.SliceThickness = slice_thickness
        
    return slices

# read the pydicom images, find HU numbers (padding, intercept, rescale), and make a 4-D array, 
def get_pixels_hu(slices):
    image = np.stack([s.pixel_array for s in slices])
    # Convert to int16 (from sometimes int16), 
    # should be possible as values should always be low enough (<32k)
    image = image.astype(np.int16)

    # Set outside-of-scan pixels to 0
    # The intercept is usually -1024, so air is approximately 0
    try:
        padding = slices[0].PixelPaddingValue
    except:
        padding = 0
    
    image[image == padding] = 0
    
    # Convert to Hounsfield units (HU)
    for slice_number in range(len(slices)):
        
        intercept = slices[slice_number].RescaleIntercept
        slope = slices[slice_number].RescaleSlope
        
        if slope != 1:
            image[slice_number] = slope * image[slice_number].astype(np.float64)
            image[slice_number] = image[slice_number].astype(np.int16)
            
        image[slice_number] += np.int16(intercept)
        
    return np.array(image, dtype=np.int16)

#  map array between zero and 1 , find max and min
def map_0_1(array):
    out = np.zeros(array.shape)
    max_out = np.zeros(array.shape[0])
    min_out = np.zeros(array.shape[0])

    for n,val in enumerate(array):
        out[n] = (val-val.min())/(val.max()-val.min())
        max_out[n] = val.max()
        min_out[n] = val.min()
    
    out = np.nan_to_num(out)

    return out.astype(np.float32)

def windowing(image,center,width):
    min_bound = center - width/2
    max_bound = center + width/2
    output = (image-min_bound)/(max_bound-min_bound)
    output[output<0]=0
    output[output>1]=1
    return output

def window_array(array, center, width):
    out = np.zeros(array.shape)
    for n,val in enumerate(array):
        out[n] = windowing(val,center,width)
    return out

#write a nmpy array in a dicom image
def write_dicom(slices,arrays,path):
    # array should be between 0-4095
    for i in range(arrays.shape[0]):
        new_slice = slices
        pixel_array = ((arrays[i,:,:]+new_slice.RescaleIntercept)/new_slice.RescaleSlope).astype(np.int16)
        pixel_array = arrays[i,:,:,0].astype(np.int16)
        new_slice.PixelData = pixel_array.tostring()
        new_slice.save_as(path+'/'+str(i)+'.dcm')
        
        

# To have similar thickness when using different datasets
def resample_thickness(image, scan, new_spacing=[1,1,1]):
    # Determine current pixel spacing
    spacing = np.array([scan[0].SliceThickness] + scan[0].PixelSpacing, dtype=np.float32)

    resize_factor = spacing / new_spacing
    new_real_shape = image.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / image.shape
    new_spacing = spacing / real_resize_factor
    
    image = scipy.ndimage.interpolation.zoom(image, real_resize_factor, mode='nearest')
    
    return image, new_spacing


#To have fixed number of slices in different datasets
#perform resmpling using spline inetrpolator for a 3D dicom images of a patient (arrays)
def resample_slices(image,num_slices=32):
    
    zoom = [num_slices/image.shape[0],1,1]
    out = scipy.ndimage.interpolation.zoom(image, zoom=zoom, mode='nearest', order=3)
    out = (np.round(out)).astype(int)
    return out
    


def extract_patches(image, patch_size=32,stride=32):
    
    images_num,h,w = image.shape
    out = np.empty((0,patch_size,patch_size))
    sz = image.itemsize
    shape = ((h-patch_size)//stride+1, (w-patch_size)//stride+1, patch_size,patch_size)
    strides = sz*np.array([w*stride,stride,w,1])

    for d in range (0,images_num):
        patches=np.lib.stride_tricks.as_strided(image[d,:,:], shape=shape, strides=strides)
        blocks=patches.reshape(-1,patch_size,patch_size)
        out=np.concatenate((out,blocks[:,:,:]))
        print(d)
    
    return out[:,:,:]

def normalization(image):
    mean_image = np.mean(image, axis = 0).astype(np.float32)
    std_image = np.std(image,axis = 0).astype(np.float32)
    out = ((image-mean_image)/std_image).astype(np.float32)
    out = np.nan_to_num(out)
    return (out,mean_image,std_image)



#first_patient = load_scan(path)
#first_patient_pixels = get_pixels_hu(first_patient)
#plt.figure()
#plt.subplot(3,1,3)
#plt.hist(first_patient_pixels.flatten(), bins=80, color='c')
#plt.xlabel("Hounsfield Units (HU)")
#plt.ylabel("Frequency")
#plt.show()
#pix_resampled, spacing = resample(first_patient_pixels, first_patient, [1,1,1])
#print("Shape before resampling\t", first_patient_pixels.shape)
#print("Shape after resampling\t", pix_resampled.shape)
## Show some slice in the middle
#plt.subplot(3,1,1),plt.imshow(first_patient_pixels[2], cmap=plt.cm.gray)
#plt.show()
#
#image = windowing(first_patient_pixels,-1000,500)
#plt.subplot(3,1,2),plt.imshow(image[2], cmap=plt.cm.gray)
#plt.show()

def display(path,center,width):
    ct = load_scan(path)
    ct_pixels = get_pixels_hu(ct)
    ct_map = map_0_1(ct_pixels)
    out = np.zeros(ct_map.shape)
    for n,val in enumerate(ct_map):
        out[n] = windowing(val*4095-1024,center,width)
        plt.figure();plt.imshow(out[n],cmap='gray');plt.title(n)
    write_dicom(ct[0],ct_map*4095+1024,
           'C://Users/Maryam/Desktop/Results')
