# -*- coding: utf-8 -*-
"""

@author: Maryam
"""
import numpy as np # linear algebra
import pydicom
import os
import scipy.ndimage
import pandas as pd
import matplotlib.pyplot as plt
from preprocess_CT_image import resample_slices , load_scan, get_pixels_hu, write_dicom, windowing
import h5py
import SimpleITK as sitk
from bbox import load_basemask



def PREDICT_output(xls_path):

    '''This function returns 0/1 for no expansion/expansion of ICH and name of the subject 
    It eliminates the NA rows'''
    df = pd.read_excel(xls_path)
    df=df.dropna(axis=0,subset=['bvoltot_rep','fvoltot_rep']) # Drop NA's 
    df.reset_index(drop=True, inplace=True)
    ICH_change = df['fvoltot_rep']-df['bvoltot_rep'] # compute change in size
    ICH_per_change = ICH_change/df['bvoltot_rep'] # compute percent change
#    df['output']=  np.where((ICH_change>6) | (ICH_per_change>0.3),1,0)
#    df['output']=  np.where(np.isnan(ICH_change) | np.isnan(ICH_per_change),np.nan,df['output'])

    outputs = np.where((ICH_change>6) | (ICH_per_change>0.3),1,0)
#    output=  np.where(np.isnan(ICH_change) | np.isnan(ICH_per_change),np.nan,output)
    subjects =r'PREDICT_' +(df['site'].map(str)).str.zfill(2)+'-'+ (df['subjectid'].map(str)).str.zfill(3)  #   
    return subjects, outputs





def predict_base_dirs(dcm_path, xls_path, mask_path, hight=512,width = 512, num_slices=32):
    '''   #Find the folders wth the subject ID and read dicom images in their base folders.
    If the base folder or patient is not available remove the output.
    First dimension shows different patients, and the fourth dimension includes slices for a patient  '''
    arrays=np.empty((0,num_slices,hight,width),dtype = 'int')
    bbox_arrays = np.empty((0,num_slices,5), dtype = 'int')

    subjects , outputs = PREDICT_output(xls_path) # read subjects and outputs from excel file
    i=0
    list_dir_base=[]
    
    for subjectid in subjects:
        dir_name = [] # generates error in path.join if not assigned later 
        try:
            
            # path to the mask, if it does not exist throw an error
            subject_id = subjectid[:7]+' '+subjectid[8:10]+'-'+subjectid[11:] # to match folder name in mask dir
            mask_subject_path = os.path.join(mask_path,subject_id)
            os.scandir(mask_subject_path)
            # list of directories in each patient's file
            for s in os.listdir(os.path.join(dcm_path,subjectid)): # generates error if the path  does not exist 
                if 'base' in s.lower():  # if there is a base folder
                    dir_name = s
                    print(subjectid)
                    slices = load_scan(os.path.join(dcm_path,subjectid,dir_name,'series')) # load all the slices in the directory list of n slices
#                    slices = load_scan(os.path.join(dcm_path,subjectid,dir_name)) # load all the slices in the directory dcm_path='D:\\server V\ichcASES\Predict'
#                    pixel_values = get_pixels_hu(slices) # find the pixel values (512,512,n)
#                    values = resample_slices(pixel_values,num_slices) # change to (512,512,num_slices)
#                    
#                    arrays = np.append(arrays,values[None,:,:,:],axis=0) # stack on a new dimension (batch, 512,512,num_slices)
##                    
            # generate the path to the baseline,
            # this line generates an error when dir_name is not assigned ( no base folder found)
            list_dir_base.append(os.path.join(dcm_path,subjectid,dir_name))
                     
            '''Find the bounding boxes for this subjectid/patient'''  

#            print(mask_subject_path)
            bbox,_ = load_basemask(mask_subject_path,slices[0],len(slices))# generate error if the folder is not available
            bbox_arrays = np.append(bbox_arrays,bbox[None,:,:],axis=0)
            
         
                    
                    
                
#           plot resampled ct images in the brain window
#            for i in range(32):
#                plt.subplot(4,8,i+1)
#                plt.imshow(windowing(values[i],40,80),cmap='gray')
#                plt.axis('off')
#                plt.subplots_adjust(wspace=0, hspace=0)
#
#                
#                plt.suptitle(subjectid)
#                plt.savefig('D://Maryam-Dataset/Predict/Complete_Predict/Images/'+subjectid+'.jpg', bbox_inches="tight")
#                                        
#                    
###                    write to dicom 
##                    new_path = os.path.join('D:/Maryam-Dataset/Predict/32-Predict',subjectid)
##                    os.mkdir(new_path)
##                    write_dicom(slices[0],values,new_path)
###                    
##               
                           
        except:
            # if the folder does not exist detelt the output and subjectid
            i=i+1
            print (i,subjectid," Patient CT or baseline CT is not available " )
            void = np.where(subjects==subjectid)
#            print(void)
            outputs = np.delete( outputs, (void), axis=0) 
            subjects = subjects.drop(axis =0, index=void[0])
            subjects.reset_index(inplace=True, drop=True)
#            
            
    return subjects, outputs, arrays,bbox_arrays, list_dir_base 





 
    
def write_hdf5(data,labels, output_filename):
    """
    This function is used to save image data and its label(s) to hdf5 file.
    output_file.h5,contain data and label
    """

    with h5py.File(output_filename, 'w') as h:
        h.create_dataset('data', data=data, shape=data.shape)
        h.create_dataset('label', data=labels, shape=labels.shape)

#        
dcm_path='D:\Maryam-Dataset\Predict\Predict_ftp\PREDICT'
#dcm_path='D:\\Maryam-Dataset\Predict\Complete_Predict'
xls_path='D:\\server V\ichcASES\PREDICT final database v4 2013.06.16.xlsx'
mask_path='D:\\Maryam-Dataset\Predict\Volume_Measurements'

#subjects, outputs, arrays,bbox_arrays,list_dir_base = predict_base_dirs(dcm_path,xls_path,mask_path)

#arrays[np.where(arrays>3071)]= 3071
#arrays[np.where(arrays<-1024)] = -1024
#
#write_hdf5(arrays,outputs, 'D:\\Maryam-Dataset\H5-files\predict_dataset.h5')
#subjects.to_hdf('D:\\Maryam-Dataset\H5-files\predict_dataset_subjects.h5', key='subjects')
#address = pd.Series( (v for v in list_dir_base) )
#address.to_hdf('D:\\Maryam-Dataset\H5-files\predict_dataset_subjects.h5', key='address')

#
#with h5py.File('D:\\Maryam-Dataset\H5-files\predict_bbox.h5', 'w') as h:
#    h.create_dataset('bbox', data=bbox_arrays, shape=bbox_arrays.shape)