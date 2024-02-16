# -*- coding: utf-8 -*-
"""
Created on Wed Feb 13 12:34:40 2019

@author: Maryam
This function load 
"""
import numpy as np # linear algebra
import pydicom
import os
import scipy.ndimage
import matplotlib.pyplot as plt
from preprocess_CT_image import resample_slices , load_scan, get_pixels_hu
import h5py
import SimpleITK as sitk

import pandas as pd


def spotlight_output(xls_path):

    '''This function returns 0/1 for no expansion/expansion of ICH and name of the subject 
    It eliminates the NA rows'''
    df = pd.read_excel(xls_path)
    df=df.dropna(axis=0,subset=['Subject','Total Growth','Percent Growth Total']) # Drop NA's 
    df.reset_index(drop=True, inplace=True)
    outputs = np.where((df['Total Growth']>6) | (df['Percent Growth Total']>0.3),1,0) # compute output
    
    subjects=(df['Subject'].map(int)).map(str).str.zfill(7)# read the subjects and add 0 to the begining, make 7 digit

    return subjects, outputs





def spotlight_dirs(dcm_path, xls_path,  hight=512,width = 512, num_slices=32):
    '''   #Find the folders wth the subject ID and read dicom images in their base folders.
    If the base folder or patient is not available remove the output.
    First dimension shows different patients, and the fourth dimension includes slices for a patient  '''
    arrays=np.empty((0,hight,width,num_slices))
 
    list_dir_base =[]

    subjects , outputs = spotlight_output(xls_path) # read subjects and outputs from excel file
    i=0
    for subjectid in subjects:
        dir_name = [] # generates error in path.join if not assigned later 
        subject_id ='-'.join([subjectid[:3],subjectid[3:]]) # insert hyphen to match the folder names
        try:
            # list of directories in each patient's file
            for s in os.listdir(os.path.join(dcm_path,subject_id)): # generates error if the path  does not exist 
#                print(subject_id,s)
                dir_name = os.path.join(dcm_path,subject_id,s)
                if os.path.isdir(os.path.join(dir_name,os.listdir(dir_name)[0])):
                    dir_name = os.path.join(dir_name, os.listdir(dir_name)[0])
                print(dir_name)
                slices = load_scan(os.path.join(dcm_path,subject_id,dir_name)) # load all the slices in the directory 
                pixel_values = get_pixels_hu(slices) # find the pixel values (512,512,n)
                values = resample_slices(pixel_values,num_slices) # change to (512,512,num_slices)
                arrays = np.append(arrays,values[None,:,:,:],axis=0) # stack on a new dimension (batch, 512,512,num_slices)
            
                
##                    write to dicom 
#                    new_path = os.path.join('D:/Maryam-Dataset/Predict/32-Predict',subjectid)
#                    os.mkdir(new_path)
#                    write_dicom(slices[0],values,new_path)
##                    
            list_dir_base.append(os.path.join(dcm_path,subjectid,dir_name)) # generate the path to the folder,
            # this line geerates error when dir_name is not assigned ( no base folder found)
                
#               
                           
        except:
            # if the folder does not exist detelt the output and subjectid
            i=i+1
            print (subjectid," Patient CT or baseline CT is not available " )
            void = np.where(subjects==subjectid)
#            print(void)
            outputs = np.delete( outputs, (void), axis=0) 
            subjects = subjects.drop(axis =0, index=void[0])
            subjects.reset_index(inplace=True, drop=True)
#            
            
    return subjects, outputs, arrays, list_dir_base


def load_dir_scans(list_dir_base, hight=512,width = 512, num_slices=32):
        
    '''load_dir_scan read the dicom files in the dir_list_base and generates an array.
    First dimension shows different patients, and the fourth dimension includes slices for a patient'''
        
    arrays=np.empty((0,hight,width,num_slices))
    
    for dir_base in list_dir_base:
        slices = load_scan(dir_base) # load all the slices in the directory list of n slices
        pixel_values = get_pixels_hu(slices) # find the pixel values (512,512,n)
        values=resample_slices(pixel_values,num_slices) # change to (512,512,num_slices)
        arrays = np.append(arrays,values[None,:,:,:],axis=0) # stack on a new dimension (batch, 512,512,num_slices)
                    
        print(arrays.shape)
        
    return arrays
                     





















def load_basemask(path,hight=512,width = 512, num_slices=32):
    # this function searches for the folders that have mask  !!!!!! This function does not work correcly 
    for dirpath, dirList, fileList in os.walk(path):
            for dirname in dirList:
                 if 'base' in dirname.lower():
                     if 'total' in dirname.lower(): # look in directories with base & total in their name
                         dir_base = os.path.join(dirpath,dirname)# in our dataset images are stored in a folder called series
                         print(dir_base)
                         break
                
                     elif 'ich' in dirname.lower(): # look in directories with base & total in their name
                         dir_base = os.path.join(dirpath,dirname)# in our dataset images are stored in a folder called series
                         print(dir_base)
                         break
                    
#                     else :
#                         dir_base = os.path.join(dirpath,dirname)# in our dataset images are stored in a folder called series
#                         print(dir_base)
                         
                    
    return arrays



def bounding_box(mask_path,ct_slices, hight=512, width=512):
    # This function finds the bounding boxes for ct-images of a patient 
    segment_itk = sitk.ReadImage(mask_path)
    # Convert the image to a  numpy array first and then
    # shuffle the dimensions to get axis in the order z,y,x
    segment_mask = sitk.GetArrayFromImage(segment_itk)
    # Read the origin of the ct_scan, will be used to convert the 
    #coordinates from world to voxel and vice versa.    
    segment_origin = np.array(segment_itk.GetOrigin())
    # Read the spacing along each dimension
    spacing = np.array(segment_itk.GetSpacing())
    
    ct_origin = np.array(ct_slices[0].ImagePositionPatient)
    
    # calculate the distance between origin of mask and CT_slice in pixel
    dx,dy,dz = (np.round(np.abs(((segment_origin-ct_origin)/spacing[1])))).astype(int)
    z,x,y = segment_mask.shape
    
    
    # If we want the mask withsize of the ct image
#    mask = np.zeros((len(ct_slices),hight,width))
#    mask[0:z,dx:dx+x,dy:dy+y] = segment_mask
#    bbox = np.where(mask==1)
    
    
    # find the bounding box corners
     # z shows the  ct scans numbers and 5 is for p,
    #row min, row max, col min, col max 
    bbox = np.zeros((z,5))
    coordinates = np.array(np.where(segment_mask==1))

    for k in np.unique(coordinates[0,:]): #find the ct-scan number 
        bbox[k,0] = 1 # there is an object in this image
        index = np.where(coordinates[0]==k)
        row_min , row_max = np.min(coordinates[1,index])+dx , np.max(coordinates[1,index])+dx
        col_min , col_max = np.min(coordinates[2,index])+dy ,  np.max(coordinates[2,index])+dy
        bbox[k,1:] = row_min , row_max, col_min , col_max
        
        # finf the center and hight and widdth of bounding box
#        bbox[k,1:] = np.round(row_min+row_max)/2,np.round(col_min+col_max)/2,row_max-row_min,col_max-col_min
    return bbox
    


 
    
def write_hdf5(data,labels, output_filename):
    """
    This function is used to save image data and its label(s) to hdf5 file.
    output_file.h5,contain data and label
    """

    # h.create_dataset()
    with h5py.File(output_filename, 'w') as h:
        h.create_dataset('data', data=data, shape=data.shape)
        #h.create_dataset('mean_data', data=mean_data, shape=mean_data.shape)
       # h.create_dataset('std_data', data=std_data, shape=std_data.shape)

        h.create_dataset('label', data=labels, shape=labels.shape)
#        h.create_dataset('mean_labels', data=mean_labels, shape=mean_labels.shape)
#        h.create_dataset('std_labels', data=std_labels, shape=std_labels.shape)
        
        
dcm_path='D:\\Maryam-Dataset\sPOTLIGHT'
xls_path='D:\\server V\ichcASES\GraebCombined_withmRS (Recovered).xlsx'
#arrays = load_baseline(path)

subjects, outputs, arrays,list_dir_base = spotlight_dirs(dcm_path,xls_path)

write_hdf5(arrays,outputs, 'D:\\Maryam-Dataset\H5-files\spotlght_dataset.h5')
subjects.to_hdf('D:\\Maryam-Dataset\H5-files\spotlight_dataset_subjects.h5', key='subjects')
address = pd.Series( (v for v in list_dir_base) )
address.to_hdf('D:\\Maryam-Dataset\H5-files\spothlight_dataset_subjects.h5', key='address')
