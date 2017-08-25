import numpy as np
import scipy as sp
import pandas as pd
import h5py
#from pyminc.volumes.factory import *
import os
from re import sub
from keras.models import Sequential, load_model
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
from shutil import copy
import argparse

# fix random seed for reproducibility
np.random.seed(8)

def set_images(source_dir,ratios):
    '''Creates a DataFrame that contains a list of all the subjects along with their PET images, T1 MRI images and labeled images.'''

    subject_dirs = glob(source_dir+os.sep+'*')
    pet_list = glob( source_dir + os.sep + '*' + os.sep + '*_pet.*'  )
    t1_list = glob( source_dir + os.sep + '*' + os.sep + '*_t1.*'  )
    label_list = glob( source_dir + os.sep + '*' + os.sep + '*_labels_brainmask.*'  )
    names = [ basename(f) for f in  subject_dirs ]
    colnames=["subject", "radiotracer", "pet", "t1", "label"]
    nSubjects=len(names)
    out=pd.DataFrame(columns=colnames)
    for name in names:
        label = [ f for f in label_list if name in f][0] 
        pet =  [ f for f in pet_list if name in f ]
        t1 =  [ f for f in t1_list if name in f ][0]
        pet_names = [ sub('acq-','', g) for f in pet for g in f.split('_') if 'acq' in g ]
        n=len(pet)
        subject_df = pd.DataFrame(np.array([[name] * n,  pet_names,pet,[t1]*n, [label] * n]).T, columns=colnames)
        out = pd.concat([out, subject_df ])

    nfolds=np.random.multinomial(nSubjects,ratios) #number of test/train subjects
    image_set = ['train'] * nfolds[0] + ['test'] * nfolds[1]
    out["category"] = "unknown"
    #out.reset_index(inplace=True)
    l = out.groupby(["subject"]).category.count().values
    # please change later
    el = []
    for i in range(nSubjects):
       el.append([image_set[i]]*l[i])
    el = [l for sublist in el for l in sublist]
    out.category = el
    return out

def generator(f, batch_size):
    i=0
    start=i*batch_size
    end=start + batch_size
    #while end < tensor_max:
    while True:
        start=i*batch_size
        end=start + batch_size #(i+1)*batch_size 
        X = f['image'][start:end,]
        Y = f['label'][start:end,]
        i+=1
        yield [X,Y]
       

def feature_extraction(images, nTrain, nTest, target_dir, batch_size, tensor_dim, image_dim, feature_dim=3 , use_patch=False, parameters=None, normalize=True, clobber=False):
    '''Extracts the features from the PET images according to option set in feature type.
    Feature type options: 
        1) Full image (no features extracted): return 3d array. Parameters = None
        2) Slice: return list of 2d image slices 
        3) Lines: return list of 1d profiles 
    '''
    nImages=images.shape[0]
    temp_dir = target_dir + os.sep + 'chunk'
    if not exists(temp_dir): os.makedirs(temp_dir)

    dim_range = int(tensor_dim[0] / nImages )
    train_dim = [ int((tensor_dim[0]/nImages)*nTrain)  ]  +  tensor_dim[1:] + [1] 
    test_dim = [ int((tensor_dim[0]/nImages)*nTest)  ]  +  tensor_dim[1:] + [1] 
    train_fn=temp_dir+os.sep+"train_batch-size-"+str(batch_size) +'_type-' + str(feature_dim)+ ".hdf5"
    test_fn=temp_dir+os.sep+"test_batch-size-"+str(batch_size) +'_type-' + str(feature_dim) + ".hdf5"
    if not exists(train_fn) and not exists(test_fn) or clobber==True :
        train_f=h5py.File(train_fn, "w")
        test_f=h5py.File(test_fn, "w") 
        
        X_train_set = train_f.create_dataset('image', train_dim, dtype='float16')
        Y_train_set = train_f.create_dataset("label", train_dim, dtype='float16')
        
        X_test_set = test_f.create_dataset("image", test_dim, dtype='float16')
        Y_test_set = test_f.create_dataset("label", test_dim, dtype='float16')
        #for each image in this chunk...
        for i in range(nImages): 
            #identify and load the corresponding pet and label images
            row=images.iloc[i, ]
            minc_pet_f = h5py.File(row.pet, 'r')
            minc_label_f = h5py.File(row.label, 'r')
            pet=np.array(minc_pet_f['minc-2.0/']['image']['0']['image']) #volumeFromFile(row.pet).data
            
            #sum the pet image if it is a 4d volume
            if len(pet.shape) == 4: pet = np.sum(pet, axis=0)
            label=np.array(minc_label_f['minc-2.0/']['image']['0']['image']) #volumeFromFile(row.label).data
            #if normalize: pet = (pet - pet.min())/(pet.max() - pet.min())
            pet=pet.reshape(list(pet.shape)+[1])
            label=label.reshape(list(label.shape)+[1])

            if images.iloc[i].category == 'train': f= train_f
            elif images.iloc[i].category == 'test': f=test_f
            
            for j in range(dim_range):
                if feature_dim ==3 : 
                    f['image'][j,...]=pet
                    f['label'][j,...]=label
                elif feature_dim ==2 :
                    f['image'][j,...] =pet[j,:,:]
                    f['label'][j,...]=label[j,:,:]
                elif feature_dim== 1 :
                    z=int(j / (image_dim[1]))
                    y=j-z*image_dim[1] 
                    f['image'][j,...]=pet[z,y,:,:]
                    f['label'][j,...]=label[z,y,:,:]
            
        train_f.close()
        test_f.close()
    return [train_fn , test_fn]



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
        model.add(Dense(1, activation="tanh"))
    elif feature_dim == 2 : 
        model.add(ZeroPadding2D(padding=(1, 1),batch_input_shape=shape,data_format="channels_last" ))
        model.add(Conv2D( 4 , [3,3],  activation='relu'))
        model.add(Dense(4))
        model.add(Dense(1))

    else  :
        model.add(ZeroPadding3D(padding=(1, 1, 1),batch_input_shape=shape,data_format="channels_last" ))
        model.add(Conv3D( 32 , [3,3,3],  activation='relu'))
        model.add(Dense(32))
        model.add(Dense(1))
    
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model
from fractions import gcd

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

def pet_brainmask_convnet(source_dir, target_dir, ratios, feature_dim=3, use_patch=False, batch_size=2, nb_epoch=10,shuffle_training=True, clobber=False, model_name=False ):

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
    if feature_dim ==3 : tensor_dim = [nImages]+image_dim
    elif feature_dim ==2 : tensor_dim = [nImages*image_dim[0]]+image_dim[1:3]
    elif feature_dim ==1 : tensor_dim = [nImages*image_dim[0]*image_dim[1]]+[image_dim[2]]
    nUnique =int( tensor_dim[0] / nImages)
    
    nSubjects= len(np.unique(images.subject)) #total number of subjects
    category_sizes = images.groupby(['category']).size()
    nTrain = int(category_sizes.ix['train','category']) *nUnique #Number of training samples
    nTest = int(category_sizes.ix['test','category']) *nUnique #Number of testing samples
    #determine the number test batches
    batch_size  = adjust_batch_size(nTrain,nTest, batch_size) #adjusted size of batch so that it divides nTrain and nTest without remainder
    input_shape= [batch_size] +  tensor_dim[1:] + [1] #input shape for base layer of network
    ### 4) Define architecture of neural network
    model = define_arch(input_shape, feature_dim)
   
    n_train_batches = int(nTrain / batch_size)
    n_test_batches = int(nTest / batch_size)
    ### 6) Take all of the subject data, extract the desired feature, store it in a tensor, and then save it to a common hdf5 file
    train_fn, test_fn = feature_extraction(images, nTrain, nTest, target_dir, batch_size, tensor_dim, image_dim, feature_dim=feature_dim,  clobber=clobber )
    #Open the hdf5 for reading
    train_f = h5py.File(train_fn, 'r')
    test_f = h5py.File(test_fn, 'r')

    ### 7) Train network on data

    if model_name == None:  model_name =target_dir+os.sep+ 'model_'+str(feature_dim)+'.hdf5' 
    if exists(model_name) :
    #If user provides a model that has already been trained, load it
        load_model(model_name)
    else :
    #If model_name does not exist, or user wishes to write over (clobber) existing model
    #then train a new model and save it
        model.fit_generator( generator(train_f, batch_size), steps_per_epoch=n_train_batches, epochs=nb_epoch, validation_data=generator(test_f, batch_size ), validation_steps=n_test_batches , max_queue_size=10, workers=1, use_multiprocessing=True )
        model.save(model_name)

    ### 8) Evaluate network #FIXME : does not work at the moment 
    #scores = model.evaluate(X_test, Y_test,batch_size=tensor_dim[0] )
    #print("Scores: %s: %.2f%%" % (model.metrics_names[1], scores[1]*100))


    ### 9) Produce prediction    
    '''start=0
    end=nUnique
    X = f['image'][start:end,]
    X_predict=model.predict(X, batch_size=1 )
    out_fn=target_dir + os.sep + sub('.mnc', '_predict.mnc', os.path.basename(label_fn))
    X_predict=X_predict.reshape(image_dim)
    if exists(out_fn) : os.remove(out_fn)

    #copy(label_fn, out_fn)
    predict_f = h5py.File(out_fn, 'w')
    print( X_predict.shape )
    dset = predict_f.create_dataset('minc-2.0/image/0/image',shape=X_predict.shape, dtype='f')
    dset[...] = X_predict
    #   predict_f['minc-2.0/']['image']['0']['image'][:] = X_predict[:]
    predict_f.close()
    '''


    return 0
   

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--batch-size', dest='batch_size', type=int, default=1, help='size of batch')
    parser.add_argument('--source', dest='source_dir', type=str, help='source directory')
    parser.add_argument('--target', dest='target_dir', type=str, help='target directory')
    parser.add_argument('--epochs', dest='nb_epoch', type=int,default=10, help='target directory')
    parser.add_argument('--feature-dim', dest='feature_dim', type=int,default=3, help='Format of features to use (3=Volume, 2=Slice, 1=profile')
    parser.add_argument('--clobber', dest='clobber',  action='store_true', default=False,  help='clobber')
    parser.add_argument('--load-model', dest='model_name', default=None,  help='clobber')
    parser.add_argument('--ratios', dest='ratios', nargs=2, type=float , default=[0.7,0.3],  help='List of ratios for training, testing, and validating (default = 0.7 0.2 0.1')
    args = parser.parse_args()
    pet_brainmask_convnet(args.source_dir, args.target_dir, feature_dim = args.feature_dim, ratios=args.ratios, batch_size=args.batch_size, nb_epoch=args.nb_epoch, clobber=args.clobber, model_name = args.model_name)
