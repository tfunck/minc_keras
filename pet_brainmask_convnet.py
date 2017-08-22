import numpy as np
import scipy as sp
import pandas as pd
from pyminc.volumes.factory import *
import os
from re import sub
from keras.models import Sequential
from keras.layers import Dense, MaxPooling3D
from keras.layers.convolutional import Conv1D, Conv2D, Conv3D
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.convolutional import ZeroPadding3D, ZeroPadding2D, ZeroPadding1D
from sys import argv, exit
from glob import glob
from os.path import basename, exists
from math import ceil
from random import shuffle
import argparse

# fix random seed for reproducibility
np.random.seed(8)

def set_csv(temp, A , feature_dim, batch, n, m) :
    return temp +os.sep+ A + '_type-'+str(feature_dim)+'_chunk-'+str(batch)+'-'+str(n)+'-'+str(m)


def set_images(source_dir):
    '''Creates a DataFrame that contains a list of all the subjects along with their PET images, T1 MRI images and labeled images.'''

    subject_dirs = glob(source_dir+os.sep+'*')
    pet_list = glob( source_dir + os.sep + '*' + os.sep + '*_pet.*'  )
    t1_list = glob( source_dir + os.sep + '*' + os.sep + '*_t1.*'  )
    label_list = glob( source_dir + os.sep + '*' + os.sep + '*_labels_brainmask.*'  )

    names = [ basename(f) for f in  subject_dirs ]
    colnames=["subject", "radiotracer", "pet", "t1", "label"]
    out=pd.DataFrame(columns=colnames)
    for name in names:
        label = [ f for f in label_list if name in f][0] 
        pet =  [ f for f in pet_list if name in f ]
        t1 =  [ f for f in t1_list if name in f ][0]
        pet_names = [ sub('acq-','', g) for f in pet for g in f.split('_') if 'acq' in g ]
        n=len(pet)
        subject_df = pd.DataFrame(np.array([[name] * n,  pet_names,pet,[t1]*n, [label] * n]).T, columns=colnames)
        out = pd.concat([out, subject_df ])
    return out



def feature_extraction(images, temp_dir, batch_size, tensor_dim, feature_dim=3, use_patch=False, parameters=None, normalize=False, clobber=False):
    '''Extracts the features from the PET images according to option set in feature type.
    Feature type options: 
        1) Full image (no features extracted): return 3d array
            Parameters = None
        2) Slice: return list of 2d image slices 
            Parameters = integer, axis (0,1,2) along which to extract slices (default=0=z=axial)
        3) Lines: return list of 1d profiles 
            Parameters = None
        4) 2D Kernel: return list of 2d patches from full image
            Parameters = integer, size of kernel 
        5) 3D Kernel: return list of 3d patches from full image
            Parameters = integer, size of kernel 
    '''

    m=images.shape[0]
    if not exists(temp_dir): os.makedirs(temp_dir)
    X_list=np.array([])
    Y_list=np.array([])   
    
    #image_dim = list(volumeFromFile(images.iloc[0,].pet).data.shape)
    #if len(image_dim) == 4: image_dim = image_dim[1:4]
    #the dimensions of the tensor we want to save as a batch depends
    #on the dimensions of the features (full volume, 2d slice, 1d profile)
    #full volume = [number of images, z dim, y dim, x dim]
    #2d slice  = [number of images * z dim, y dim, x dim]
    #1d profile = [number of images * z dim * y dim, x dim]
    #note: batch_size = number of images in the batch


    tensor_samples = int(tensor_dim[0])
    dim_range = int(tensor_samples / batch_size)

    for b in np.arange(0, m, batch_size, dtype=int):
        #chunks represents the chunks of image data that will go in a single batch
        chunks = np.arange(b, b + batch_size )
        chunks = chunks[ chunks < m ]

        X_csv=set_csv(temp_dir,'X', feature_dim, chunks[0], batch_size, m)
        Y_csv=set_csv(temp_dir,'Y', feature_dim, chunks[0], batch_size, m)
        X_list = np.append(X_list, [X_csv+'.npy'])
        Y_list = np.append(Y_list, [Y_csv+'.npy'])
        if not exists(X_csv+'.npy') or not exists(Y_csv+'.npy') or clobber==True :
            #for each image in this chunk...
            for i in chunks:
                #identify and load the corresponding pet and label images
                row=images.iloc[i, ] 
                pet=volumeFromFile(row.pet).data
                label=volumeFromFile(row.label).data
                #allocate the tensors in which we will store the chunk data
                try : X
                except NameError :X= np.zeros(tensor_dim) 

                try : Y
                except : Y= np.zeros(tensor_dim) 
                
                #sum the pet image if it is a 4d volume
                if len(pet.shape) == 4: pet = np.sum(pet, axis=0)
                #normalize the pet volume between 0 and 1
                if normalize==True: pet = (pet - pet.min()) / (pet.max() -pet.min())

                for j in range(dim_range):
                    if feature_dim ==3 : 
                        X[j]=pet
                        Y[j]=label
                    elif feature_dim ==2 :
                        X[j]=pet[j,:,:]
                        Y[j]=label[j,:,:]
                    elif feature_dim==1:
                        z=k % image_shape[1]
                        y=k % image_shape[2]
                        X[j]=pet[z,y,:]
                        Y[j]=label[z,y,:]

            X=X.reshape( list(X.shape) + [1] )
            Y=Y.reshape( list(Y.shape) + [1] )
            np.save(X_csv, X)
            np.save(Y_csv, Y)
            del X
            del Y


    return [X_list, Y_list] 



def define_arch(shape,feature_dim=3):
    '''Define architecture of neural net'''
    # create model
    model = Sequential()
    if feature_dim == 1 : 
        model.add(Conv1D( 32, 3, activation='relu',input_shape=shape))
    elif feature_dim == 2 : 
        model.add(ZeroPadding2D(padding=(1, 1),batch_input_shape=shape,data_format="channels_last" ))
        model.add(Conv2D( 1 , [3,3],  activation='relu'))
        model.add(Dense(1))
        model.add(Dense(1))

    else  :
        model.add(ZeroPadding3D(padding=(1, 1, 1),batch_input_shape=shape,data_format="channels_last" ))
        model.add(Conv3D( 32 , [3,3,3],  activation='relu'))
        model.add(Dense(32))
        model.add(Dense(1))
    
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model

def pet_brainmask_convnet(source_dir, target_dir, ratios, feature_dim=3, use_patch=False, batch_size=2, nb_epoch=1,shuffle_training=True, clobber=False ):
    #1) Organize inputs into a data frame, match each PET image with label image
    images = set_images(source_dir)
    #2) Set up dimensions of tensors to be used for training and testing
    image_dim =  volumeFromFile(images.iloc[0].label).sizes[0:3] 
    if feature_dim ==3 : tensor_dim = [batch_size]+image_dim
    elif feature_dim ==2 : tensor_dim = [batch_size*image_dim[0]]+image_dim[1:3]
    elif feature_dim ==1 : tensor_dim = [batch_size*image_dim[0]*image_dim[1]]+image_dim[2]
    input_shape=  tensor_dim + [1]
    #3) Define the inputs and outputs to system archetecture
    print (input_shape)
    model = define_arch(input_shape, feature_dim)

    nImages = images.shape[0]
    nbatches = ceil(nImages / batch_size)
    nfolds=np.random.multinomial(nbatches,ratios)
    total_folds = sum(nfolds)
    temp_dir = target_dir + os.sep + 'chunk'
    X_list, Y_list = feature_extraction(images, temp_dir, batch_size, tensor_dim,  feature_dim=feature_dim, normalize=True, clobber=clobber )
    for fold in range(total_folds):
        print ('Fold =', fold)
        train_i = (fold + np.arange(0, nfolds[0])) % nbatches
        test_i =  (fold + np.arange(nfolds[0], nfolds[0] + nfolds[1])) % nbatches
       
        if shuffle_training : 
            shuffle(train_i)
        
        X_train_list=X_list[ train_i  ]
        Y_train_list=Y_list[ train_i ]
        X_test_list=X_list[ test_i ]
        Y_test_list=Y_list[ test_i ]
        
        for e in range(nb_epoch):
            #for i in np.arange(0, n, batch_size, dtype=int):
            print(e)
            for X_csv, Y_csv in zip(X_train_list, Y_train_list):
                X_train = np.load(X_csv)
                Y_train = np.load(Y_csv)
                print(X_train.shape)
                print(Y_train.shape)
                model.fit(X_train, Y_train, batch_size=tensor_dim[0],  nb_epoch=1)
        
        for X_csv, Y_csv in zip(X_test_list, Y_test_list):
            X_test = np.load(X_csv)
            Y_test = np.load(Y_csv)
            scores = model.evaluate(X_test, Y_test)
            print("Scores: %s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
    
    return 0


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=1, help='size of batch')
    parser.add_argument('--source', dest='source_dir', type=str, help='source directory')
    parser.add_argument('--target', dest='target_dir', type=str, help='target directory')
    parser.add_argument('--epochs', dest='nb_epoch', type=int,default=10, help='target directory')
    parser.add_argument('--feature-dim', dest='feature_dim', type=int,default=3, help='Format of features to use (3=Volume, 2=Slice, 1=profile')
    parser.add_argument('--clobber', dest='clobber',  action='store_true', default=False,  help='clobber')
    parser.add_argument('--ratios', dest='ratios', nargs=3, type=float , default=[0.7,0.2,0.1],  help='List of ratios for training, testing, and validating (default = 0.7 0.2 0.1')
    args = parser.parse_args()
    pet_brainmask_convnet(args.source_dir, args.target_dir, feature_dim = args.feature_dim, ratios=args.ratios, batch_size=args.batch_size, nb_epoch=args.nb_epoch, clobber=args.clobber)
