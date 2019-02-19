import numpy as np
import scipy as sp
import pandas as pd
import h5py
#from pyminc.volumes.factory import *
import os
from re import sub
from sys import argv, exit
from os.path import basename, exists, splitext
from os import makedirs
from set_images import *
from utils import *

def adjust_batch_size(n1, n2, batch_size):
    #This little bit of code changes the batch_size so that it divides the first dimension
    #of the data tensor without remainder. This way the data tensor can be divided into 
    #equally sized batche
    n = n1
    if n > n2:n=n2
    if n1 % batch_size != 0 and n2 % batch_size != 0:
        for b in range(n, 0, -1):
            if n1 % b == 0 and n2 % b == 0:
                return b
    else: return batch_size

def pad(x, n):
    while (x % 2**n) != 0 : 
        x += (x % 2**n)
    return x

def feature_extraction(images, temp_image_dim, pad_image_dim, x_output_file, y_output_file,data_dir, clobber, pad_base=0):
    nSubjects= images.shape[0] #total number f subjects
    total_valid_slices = images.valid_samples.values.sum()
    #Set up the number of valid slices for each subject
    
    if pad_base % 1 != 0 or pad_base < 0 :
        print("Error: <pad_base> must be a integer great than or equal to 0.")
        exit(1)

    f = h5py.File(data_dir+os.sep+'temp.hdf5', "w")
    X_f = f.create_dataset("image", [total_valid_slices,pad_image_dim[1],pad_image_dim[2],1], dtype='float16')
    Y_f = f.create_dataset("label", [total_valid_slices,pad_image_dim[1],pad_image_dim[2],1], dtype='float16')
    total_index=0
    for index, row in images.iterrows():
        if index % 10 == 0: print("Saving",images["category"][0],"images:",index, '/', images.shape[0] , end='\r') 
        #meera
        #these 4 lines need to be changed
        #i think you would just need to use
        #pet=safe_h5py_open(row.pet, 'r')
        #label=safe_h5py_open(row.label, 'r')
        minc_pet_f = safe_h5py_open(row.pet, 'r')
        minc_label_f = safe_h5py_open(row.label, 'r')
        pet=np.array(minc_pet_f['minc-2.0/']['image']['0']['image']) 
        label=np.array(minc_label_f['minc-2.0/']['image']['0']['image']) 
        #meera
        #sum pet image if it is a 4d volume
        #as mentioned in the set_images.py script
        #we need to figure out which dimension refers to time when you load
        #an array with nibabel. i think it's the last (ie., 3), but I'm not 100% sure
        #in that case we would use
        #time_dimension=3

        time_dimension=0
        if len(pet.shape) == 4: pet = np.sum(pet, axis=time_dimension)
        pet = normalize(pet)
        offset1=pad_image_dim[1]-temp_image_dim[1]
        offset2=pad_image_dim[2]-temp_image_dim[2]

        pet = np.pad(pet, ((0,0),(0,offset1 ),(0, offset2)), "constant")
        label = np.pad(label, ((0,0),(0, offset1),(0, offset2)), "constant")
        pet=pet.reshape(list(pet.shape)+[1])

        for i,j in zip(np.unique(label), range(len(np.unique(label)))):
            label[ label == i ] = j
        label=label.reshape(list(label.shape)+[1])
        for j in range(row.total_samples):
            if pet[j].sum() != 0 :
                f['image'][(total_index)] = pet[j,:,:]
                f['label'][(total_index)] = label[j,:,:]
                total_index += 1

    clean_X = f['image']
    clean_Y = f['label']
    np.save(x_output_file,clean_X)
    np.save(y_output_file,clean_Y)
    f.close()
    print("")
    return( 0  )



def set_onehot(images, filename):
    onehot = np.array([])
    for i, nsamples in zip(images.onehot, images.valid_samples): 
        onehot=np.concatenate([onehot, np.repeat(i, nsamples)] )
    np.save(filename, onehot)
    return(0)

def get_image_dim(fn):
    '''get spatial dimensions for input images
    fn -- filename 
    '''
    minc_label_f = safe_h5py_open(fn, 'r')
    label_img = np.array(minc_label_f['minc-2.0/']['image']['0']['image'])
    image_dim = list(label_img.shape) #load label file and get its dimensions
    
    del label_img
    return image_dim

# Go to the source directory and grab the relevant data. Convert it to numpy arrays named validate- and train-
def prepare_data(source_dir, data_dir, report_dir, input_str, label_str, ratios=[0.75,0.15], batch_size=2, feature_dim=2, images_fn=None,  clobber=False, pad_base=0):
    data={}
    ### 0) Setup file names and output directories
    data["train_x_fn"] = data_dir + os.sep + 'train_x'

    data["train_y_fn"] = data_dir + os.sep + 'train_y'
    data["validate_x_fn"] = data_dir + os.sep + 'validate_x'

    data["validate_y_fn"] = data_dir + os.sep + 'validate_y'
    data["test_x_fn"] = data_dir + os.sep + 'test_x'

    data["test_y_fn"] = data_dir + os.sep + 'test_y'

    if images_fn==None :images_fn= report_dir+os.sep+'images.csv'
    ### 1) Organize inputs into a data frame, match each PET image with label image
    
    if not exists(images_fn) or clobber: 
        ### set_images is a very important function that will find all the PET images and their
        ### corresponding labelled images from source_dir. This function uses <input_str> and <label_str>
        ### to identify which files are inputs and labeles, respectively. The images use the BIDS file format
        ### where subject, session, task, radiotracer are specificied in the filename. These variables are parsed
        ### from the filenames and also stored in the data frame
        images = set_images(source_dir, ratios,images_fn, input_str, label_str )
    else: 
        images = pd.read_csv(images_fn)
        
    ## 2) Split images into training and validate data frames
    train_images = images[images['category']=='train'].reset_index()
    validate_images = images[images['category']=='validate'].reset_index()
    test_images = images[images['category']=='test'].reset_index()
    train_valid_samples = train_images.valid_samples.values.sum()  
    validate_valid_samples  =  validate_images.valid_samples.values.sum()

    ### 3) Get spatial dimensions of images 
    temp_image_dim = get_image_dim(images.iloc[0].label)
    data["image_dim"] = [ temp_image_dim[0], pad( temp_image_dim[1], pad_base), pad(temp_image_dim[2], pad_base) ]

    print("1",temp_image_dim)
    print("2",data["image_dim"])

    ### 4) Set up dimensions of data tensors to be used for training and validateing. all of the
    if not exists(data["train_x_fn"] + '.npy') or not exists( data["train_y_fn"] + '.npy') or clobber:
        feature_extraction(train_images,temp_image_dim, data["image_dim"], data["train_x_fn"], data["train_y_fn"], data_dir, clobber, pad_base=pad_base)
    if not exists(data["validate_x_fn"] + '.npy') or not exists(data["validate_y_fn"] + '.npy') or clobber:
        feature_extraction(validate_images,temp_image_dim, data["image_dim"], data["validate_x_fn"], data["validate_y_fn"], data_dir, clobber, pad_base=pad_base)
    if not exists(data["test_x_fn"] + '.npy') or not exists(data["test_y_fn"] + '.npy') or clobber:
        feature_extraction(test_images, temp_image_dim, data["image_dim"], data["test_x_fn"], data["test_y_fn"], data_dir, clobber)
    
    data["batch_size"] = adjust_batch_size(train_valid_samples, validate_valid_samples, batch_size)
    return [ images, data ] 
