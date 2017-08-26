import numpy as np
import scipy as sp
import pandas as pd
import h5py
#from pyminc.volumes.factory import *
import os
from re import sub
from sys import argv, exit
from glob import glob
from os.path import basename, exists
from os import makedirs
from set_images import *

def feature_extraction(images, samples_per_subject, x_output_file, y_output_file,target_dir, clobber):
    nSubjects= images.shape[0] #total number f subjects
    total_slices = nSubjects * samples_per_subject
    if not exists(x_output_file+'.npy') or not exists(y_output_file+'.npy') or clobber:
        f = h5py.File(target_dir+os.sep+'temp.hdf5', "w")
        X_f = f.create_dataset("image", [total_slices,217,181,1], dtype='float16')
        Y_f = f.create_dataset("label", [total_slices,217,181,1], dtype='float16')
        for index, row in images.iterrows():
            if index % 100 == 0: print("Saving images:", row.category, '--', 100. * float(index)/total_slices,'%', end='\r') 
            minc_pet_f = h5py.File(row.pet, 'r')
            minc_label_f = h5py.File(row.label, 'r')
            pet=np.array(minc_pet_f['minc-2.0/']['image']['0']['image']) #volumeFromFile(row.pet).data
                
            #sum the pet image if it is a 4d volume
            if len(pet.shape) == 4: pet = np.sum(pet, axis=0)
            label=np.array(minc_label_f['minc-2.0/']['image']['0']['image']) #volumeFromFile(row.label).data
            #pet = (pet - pet.min())/(pet.max() - pet.min())
            pet=pet.reshape(list(pet.shape)+[1])

            label = (label - label.min())/(label.max() - label.min())
            label=label.reshape(list(label.shape)+[1])
            for j in range(samples_per_subject):
                f['image'][(index*samples_per_subject+j)] = pet[j]
                f['label'][(index*samples_per_subject+j)] = label[j]
        index_bad_X = []
        for i in range(f['image'].shape[0]):
            X = f['image'][i]
            if X.sum() == 0:
                index_bad_X.append(i)
        
        index_bad_Y = []
        for i in range(f['label'].shape[0]):
            Y = f['label'][i]
            if Y.sum() == 0:
                index_bad_Y.append(i)
        
        not_index_bad_X = [i for i in range(f['image'].shape[0]) if i not in index_bad_X]
        #not_index_bad_Y = [i for i in range(f['label'].shape[0]) if i not in index_bad_Y]
        clean_X = f['image'][not_index_bad_X]
        clean_Y = f['label'][not_index_bad_X]
        np.save(x_output_file,clean_X)
        np.save(y_output_file,clean_Y)

        f.close()
        print('')
        return( len([not_index_bad_X])  )



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

def get_n_slices(train_fn, test_fn):
    #open .npy files for train and test so that we can find how long they are
    #not very elegant...
    X_train = np.load(prepare_data.train_x_fn+'.npy')
    X_test = np.load(prepare_data.test_x_fn+'.npy')
    nTrain=X_train.shape[0] 
    nTest =X_test.shape[0]
    del X_train
    del X_test
    return([nTrain,nTest])

# Go to the source directory and grab the relevant data. Convert it to numpy arrays named test- and train-
def prepare_data(source_dir, target_dir, ratios, batch_size, feature_dim=2, clobber=False):
    ### 1) Organize inputs into a data frame, match each PET image with label image
    images = set_images(source_dir, ratios)

    ### 2) 
    label_fn=images.iloc[0].label #get the filename for first label file
    minc_label_f = h5py.File(label_fn, 'r')
    label_img = np.array(minc_label_f['minc-2.0/']['image']['0']['image'])
    image_dim = list(label_img.shape) #load label file and get its dimensions
    nImages = images.shape[0] #the number of images is the number of rows in the images dataframe

    ### 3) Set up dimensions of data tensors to be used for training and testing. all of the
    #data that we will use for training with be stored here.
    if feature_dim ==3 : #tensor_dim = [nImages]+image_dim
        samples_per_subject = 1
    elif feature_dim ==2 : #tensor_dim = [nImages*image_dim[0]]+image_dim[1:3]
        samples_per_subject = image_dim[0]
    elif feature_dim ==1 : #tensor_dim = [nImages*image_dim[0]*image_dim[1]]+[image_dim[2]]
        samples_per_subject = image_dim[0]*image_dim[1]

    train_images = images[images['category']=='train'].reset_index()
    test_images = images[images['category']=='test'].reset_index()
    
    train_total_images = train_images.shape[0]
    test_total_images = test_images.shape[0]
    
    data_dir = target_dir + 'data' + os.sep
    if not exists(data_dir): makedirs(data_dir)

    prepare_data.train_x_fn = data_dir + os.sep + 'train_x'
    prepare_data.train_y_fn = data_dir + os.sep + 'train_y'
    prepare_data.test_x_fn = data_dir + os.sep + 'test_x'
    prepare_data.test_y_fn = data_dir + os.sep + 'test_y'
    feature_extraction(train_images, samples_per_subject, prepare_data.train_x_fn, prepare_data.train_y_fn, target_dir, clobber)
    feature_extraction(test_images, samples_per_subject, prepare_data.test_x_fn, prepare_data.test_y_fn, target_dir, clobber)

    train_total_samples, test_total_samples = get_n_slices(prepare_data.train_x_fn, prepare_data.test_x_fn)
    prepare_data.samples_per_subject = int((train_total_samples+test_total_samples) / (train_total_images + test_total_images))
    prepare_data.batch_size = adjust_batch_size(train_total_samples, test_total_samples, batch_size)
    return images 
