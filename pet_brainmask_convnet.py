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
from make_and_run_model import *

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
    print('multinomial', nfolds, ratios)
    image_set = ['train'] * nfolds[0] + ['test'] * nfolds[1]
    out["category"] = "unknown"
    out.reset_index(inplace=True)
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
       
def feature_extraction(images, samples_per_subject, x_output_file, y_output_file,target_dir, clobber):
    nSubjects= images.shape[0] #total number f subjects
    total_slices = nSubjects * samples_per_subject
    print(total_slices)
    if not exists(x_output_file+'.npy') or not exists(y_output_file+'.npy') or clobber:
        f = h5py.File(target_dir+os.sep+'temp.hdf5', "w")
        X_f = f.create_dataset("image", [total_slices,217,181,1], dtype='float16')
        Y_f = f.create_dataset("label", [total_slices,217,181,1], dtype='float16')
        for index, row in images.iterrows():
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

def define_arch(shape,feature_dim=2):
    '''Define architecture of neural net'''
    # create model
    model = Sequential()
    if feature_dim == 1 : 
        model.add(ZeroPadding1D(padding=(1),batch_input_shape=shape))
        model.add(Conv1D( 16, 3, activation='relu',input_shape=shape))
        model.add(Dense(16))
        model.add(Dense(1, activation="tanh"))
    elif feature_dim == 2 : 
        #model.add(ZeroPadding2D(padding=(1, 1),batch_input_shape=shape,data_format="channels_last" ))
        model.add(Conv2D( 16 , [3,3],  activation='relu', batch_input_shape=shape,data_format="channels_last", padding='same' ))
        model.add(Conv2D( 32 , [3,3],  activation='relu', padding='same'))
        model.add(Conv2D( 1 , [3,3],  activation='sigmoid', padding='same'))

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

# Go to the source directory and grab the relevant data. Convert it to numpy arrays named test- and train-
def prepare_data(source_dir, target_dir, ratios, feature_dim=2, clobber=False):
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
    samples_per_subject =int( tensor_dim[0] / nImages)

    train_n = images[images['category']=='train'].reset_index()
    test_n = images[images['category']=='test'].reset_index()
    prepare_data.train_x_fn = target_dir + os.sep + 'train_x'
    prepare_data.train_y_fn = target_dir + os.sep + 'train_y'
    prepare_data.test_x_fn = target_dir + os.sep + 'test_x'
    prepare_data.test_y_fn = target_dir + os.sep + 'test_y'
    feature_extraction(train_n, samples_per_subject, prepare_data.train_x_fn, prepare_data.train_y_fn, target_dir, clobber)
    feature_extraction(test_n, samples_per_subject, prepare_data.test_x_fn, prepare_data.test_y_fn, target_dir, clobber)
    return tensor_dim


def pet_brainmask_convnet(source_dir, target_dir, ratios, feature_dim=2, batch_size=2, nb_epoch=10, clobber=False, model_name=False ):
    tensor_dim = prepare_data(source_dir, target_dir, ratios, feature_dim, clobber)

    ### 1) Define architecture of neural network
    model = make_model(batch_size)

    ### 2) Train network on data
    if model_name == None:  model_name =target_dir+os.sep+ 'model_'+str(feature_dim)+'.hdf5' 
    if exists(model_name) and not clobber:
    #If user provides a model that has already been trained, load it
        load_model(model_name)
    else :
    #If model_name does not exist, or user wishes to write over (clobber) existing model
    #then train a new model and save it
        X_train=np.load(prepare_data.train_x_fn+'.npy')
        Y_train=np.load(prepare_data.train_y_fn+'.npy')
        X_test=np.load(prepare_data.test_x_fn+'.npy')
        Y_test=np.load(prepare_data.test_y_fn+'.npy')
        model = compile_and_run(model, X_train, Y_train, X_test, Y_test, batch_size)
        model.save(model_name)

    ### 8) Evaluate network #FIXME : does not work at the moment 
    # scores = model.evaluate(X_test, Y_test,batch_size=tensor_dim[0] )
    #print("Scores: %s: %.2f%%" % (model.metrics_names[1], scores[1]*100))


    ### 9) Produce prediction    
    '''start=0
    end=samples_per_subject
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

def predict(model_name, predict_directory, target_dir):
    model = None
    if exists(model_name) :
        model = load_model(model_name)
        print("Model successfully loaded", model)

    prepare_data(predict_directory, target_dir, ratios = [0.0, 1.0], feature_dim = 2, clobber = False)
    print("Data prepared")
    X_test = np.load(prepare_data.test_x_fn + '.npy')
    X_predict = model.predict(X_test, batch_size = 1)
    print("Prediction completed")
   

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--batch-size', dest='batch_size', type=int, default=1, help='size of batch')
    parser.add_argument('--source', dest='source_dir', type=str, help='source directory')
    parser.add_argument('--target', dest='target_dir', type=str, help='target directory')
    parser.add_argument('--epochs', dest='nb_epoch', type=int,default=10, help='number of training epochs')
    #parser.add_argument('--feature-dim', dest='feature_dim', type=int,default=2, help='Format of features to use (3=Volume, 2=Slice, 1=profile')
    parser.add_argument('--clobber', dest='clobber',  action='store_true', default=False,  help='clobber')
    parser.add_argument('--load-model', dest='model_name', default=None,  help='clobber')
    parser.add_argument('--ratios', dest='ratios', nargs=2, type=float , default=[0.7,0.3],  help='List of ratios for training, testing, and validating (default = 0.7 0.2 0.1')
    parser.add_argument('--predict', dest='predict', type=str, help='directory with data for prediction', default=None)
    args = parser.parse_args()
    if args.predict:
        predict(args.model_name, args.predict, args.target_dir)
    else:
        pet_brainmask_convnet(args.source_dir, args.target_dir, ratios=args.ratios, batch_size=args.batch_size, nb_epoch=args.nb_epoch, clobber=args.clobber, model_name = args.model_name)
