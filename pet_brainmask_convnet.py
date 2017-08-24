import numpy as np
import scipy as sp
import pandas as pd
import h5py
from pyminc.volumes.factory import *
import os
from re import sub
from keras.models import Sequential
from keras.layers import Dense, MaxPooling3D
from keras.layers.convolutional import Conv1D, Conv2D, Conv3D
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.convolutional import ZeroPadding3D, ZeroPadding2D, ZeroPadding1D
from keras.layers.core import Dropout
from sys import argv, exit
from glob import glob
from os.path import basename, exists
from math import ceil
from random import shuffle
import argparse

# fix random seed for reproducibility
np.random.seed(8)

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

def generator(f, batch_size, tensor_max):
    i=0
    start=i*batch_size
    end=start + batch_size
    while end < tensor_max:
        start=i*batch_size
        end=start + batch_size #(i+1)*batch_size 
        X = f['image'][start:end,]
        Y = f['label'][start:end,]
        i+=1
        yield [X,Y]
       

def feature_extraction(images, target_dir, batch_size, tensor_dim, image_dim, feature_dim=3, use_patch=False, parameters=None, normalize=True, clobber=False):
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

    nImages=images.shape[0]
    temp_dir = target_dir + os.sep + 'chunk'
    if not exists(temp_dir): os.makedirs(temp_dir)


    tensor_samples = int(tensor_dim[0])
    dim_range = int(tensor_samples / nImages )
    tensor_dim = tensor_dim + [1] 
    slice_dim = [dim_range] + tensor_dim[1:]  
    out_fn = temp_dir +os.sep + "image_label_batch-size-"+str(batch_size) +'_type-' + str(feature_dim) + ".hdf5"

    maxshape = [None,] + slice_dim[1:] 


    if not exists(out_fn) or clobber==True :
        f=h5py.File(out_fn, "w")
        X_set = f.create_dataset("image", shape=slice_dim , maxshape=tensor_dim, dtype='f')
        Y_set = f.create_dataset("label", shape=slice_dim , maxshape=tensor_dim, dtype='f')
        #for each image in this chunk...
        for i in range(nImages): 
            #identify and load the corresponding pet and label images
            row=images.iloc[i, ] 
            pet=volumeFromFile(row.pet).data
            label=volumeFromFile(row.label).data
            if normalize: pet = (pet - pet.min())/(pet.max() - pet.min())
            if len(pet.shape) == 4: pet = np.sum(pet, axis=0)
            pet=pet.reshape(list(pet.shape)+[1])
            label=label.reshape(list(label.shape)+[1])

            #allocate the tensors in which we will store the chunk data
            try : X
            except NameError :X= np.zeros(slice_dim) 

            try : Y
            except NameError : Y= np.zeros(slice_dim) 
            
            #sum the pet image if it is a 4d volume
            for j in range(dim_range):
                if feature_dim ==3 : 
                    X[j]=pet
                    Y[j]=label
                elif feature_dim ==2 :
                    X[j]=pet[j,:,:]
                    Y[j]=label[j,:,:]
                elif feature_dim==1:
                    z=int(j / (image_dim[1]))
                    y=j-z*image_dim[1] 
                    X[j]=pet[z,y,:,:]
                    Y[j]=label[z,y,:,:]
            row_count = i*slice_dim[0]
            X_set[row_count:]=X
            X_set.resize( (i+1)*slice_dim[0], axis=0)
            Y_set[row_count:]=Y
            Y_set.resize( (i+1)*slice_dim[0], axis=0)
            del X
            del Y

    f.close()
    return out_fn 



def define_arch(shape,feature_dim=3):
    '''Define architecture of neural net'''
    # create model
    model = Sequential()
    if feature_dim == 1 : 
        model.add(ZeroPadding1D(padding=(1),batch_input_shape=shape))
        model.add(Conv1D( 16, 3, activation='relu',input_shape=shape))
        #model.add(ZeroPadding1D(padding=(1)))
        #model.add(Conv1D( 16, 3, activation='relu'))
        #model.add(Dropout(0.2))
        model.add(Dense(16))
        model.add(Dense(1, activation="softmax"))
    elif feature_dim == 2 : 
        model.add(ZeroPadding2D(padding=(1, 1),batch_input_shape=shape,data_format="channels_last" ))
        model.add(Conv2D( 16 , [3,3],  activation='relu'))
        model.add(Dense(16))
        model.add(Dense(1))

    else  :
        model.add(ZeroPadding3D(padding=(1, 1, 1),batch_input_shape=shape,data_format="channels_last" ))
        model.add(Conv3D( 32 , [3,3,3],  activation='relu'))
        model.add(Dense(32))
        model.add(Dense(1))
    
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model
from fractions import gcd

def pet_brainmask_convnet(source_dir, target_dir, ratios, feature_dim=3, use_patch=False, batch_size=2, nb_epoch=10,shuffle_training=True, clobber=False ):
    #1) Organize inputs into a data frame, match each PET image with label image
    images = set_images(source_dir)
    #2) 
    label_fn=images.iloc[0].label #get the filename for first label file
    image_dim =  volumeFromFile(label_fn).sizes[0:3] #load label file and get its dimensions
    nImages = images.shape[0] #the number of images is the number of rows in the images dataframe

    #2) Set up dimensions of tensors to be used for training and testing
    if feature_dim ==3 : tensor_dim = [nImages]+image_dim
    elif feature_dim ==2 : tensor_dim = [nImages*image_dim[0]]+image_dim[1:3]
    elif feature_dim ==1 : tensor_dim = [nImages*image_dim[0]*image_dim[1]]+[image_dim[2]]
    nbatches = ceil(tensor_dim[0] / batch_size)
    nUnique = tensor_dim[0] / nImages
    input_shape= [batch_size] +  tensor_dim[1:] + [1]

    #3) Define architecture of neural network
    model = define_arch(input_shape, feature_dim)

    #4) Determine number of batches for training, testing, validating
    nfolds=np.random.multinomial(nbatches,ratios)
    total_folds = sum(nfolds)

    #5) Take all of the subject data, extract the desired feature, store it in a tensor, and then save it to a common hdf5 file
    out_fn = feature_extraction(images, target_dir, batch_size, tensor_dim, image_dim, feature_dim=feature_dim,  clobber=clobber )
    
    #Open the hdf5 for reading
    f = h5py.File(out_fn, 'r')

    #6) Train network on data
    #FIXME : keep getting <Error: StopIteration>, probably because the batch_size does not divide the first dimension of the tensor with all of the data (i.e., tensor_dim[0] % batch_size != 0)
    model.fit_generator( generator(f, batch_size, tensor_dim[0] ), steps_per_epoch=nbatches, epochs=nb_epoch,  max_queue_size=10, workers=1, use_multiprocessing=True )
    
    #7) Evaluate network #FIXME : does not work at the moment 
    #scores = model.evaluate(X_test, Y_test,batch_size=tensor_dim[0] )
    #print("Scores: %s: %.2f%%" % (model.metrics_names[1], scores[1]*100))


    #8) Produce prediction #FIXME : does not work at the moment    
    #X_predict=model.predict(X_test, batch_size=tensor_dim[0] )
    #out_fn=target_dir + os.sep + sub('.mnc', '_predict.mnc', os.path.basename(label_fn))
    #X_predict=X_predict.reshape(image_dim)
    #if exists(out_fn) : os.remove(out_fn)
    #outfile = volumeLikeFile(label_fn, out_fn)
    #outfile.data = X_predict
    #outfile.writeFile()
    #outfile.closeVolume()


    return 0
   

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--batch-size', dest='batch_size', type=int, default=1, help='size of batch')
    parser.add_argument('--source', dest='source_dir', type=str, help='source directory')
    parser.add_argument('--target', dest='target_dir', type=str, help='target directory')
    parser.add_argument('--epochs', dest='nb_epoch', type=int,default=10, help='target directory')
    parser.add_argument('--feature-dim', dest='feature_dim', type=int,default=3, help='Format of features to use (3=Volume, 2=Slice, 1=profile')
    parser.add_argument('--clobber', dest='clobber',  action='store_true', default=False,  help='clobber')
    parser.add_argument('--ratios', dest='ratios', nargs=3, type=float , default=[0.7,0.2,0.1],  help='List of ratios for training, testing, and validating (default = 0.7 0.2 0.1')
    args = parser.parse_args()
    pet_brainmask_convnet(args.source_dir, args.target_dir, feature_dim = args.feature_dim, ratios=args.ratios, batch_size=args.batch_size, nb_epoch=args.nb_epoch, clobber=args.clobber)
