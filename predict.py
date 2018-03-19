import numpy as np
import scipy as sp
import pandas as pd
from os.path import basename, exists, splitext
from os import makedirs
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from re import sub
from keras.models import  load_model
from prepare_data import * 
from glob import glob
from utils import *
import json
import argparse
from keras.utils import to_categorical
from keras.utils.generic_utils import get_custom_objects
get_custom_objects().update({"dice_metric": dice_metric})


def save_image(X_validate, X_predict, Y_validate ,output_fn, slices=None, nslices=25 ):
    '''
        Writes X_validate, X_predict, and Y_validate to a single png image. Unless specific slices are given, function will write <nslices> evenly spaced slices. 
        
        args:
        X_validate -- slice of values input to model
        X_predict -- slice of predicted values based on X_validate
        Y_validate -- slice of predicted values
        output_fn -- filename of output png file
        slices -- axial slices to save to png file, None by default
        nslices -- number of evenly spaced slices to save to png

        returns: 0
    '''
    #if no slices are defined by user, set slices to evenly sampled slices along entire number of slices in 3d image volume
    if slices == None : slices = range(0,  X_validate.shape[0], int(X_validate.shape[0]/nslices) )
    #set number of rows and columns in output image. currently, sqrt() means that the image will be a square, but this could be changed if a more vertical orientation is prefered
    ncol=int(np.sqrt(nslices))
    nrow=ncol
    fig = plt.figure(1 )
    #using gridspec because it seems to give a bit more control over the spacing of the images. define a nrow x ncol grid
    #outer_grid = gridspec.GridSpec(nrow, ncol,wspace=0.0, hspace=0.0)
    slice_index=0 #index value for <slices>
    #iterate over columns and rows:
    for col in range(ncol):
        for row in range(nrow) :
            s=slices[slice_index]
            i=col*nrow+row+1
            
            #normalize the three input numpy arrays. normalizing them independently is necessary so that they all have the same scale
            A=normalize(X_validate[s])
            B=normalize(Y_validate[s])
            C=normalize(X_predict[s])

            #print("\t\t", X_validate[s].max(), X_validate[s].min() , Y_validate[s].max(), Y_validate[s].min(), X_predict[s].max(), X_predict[s].min(),end='')
            #print("\t\t", A.max(), A.min(), B.max(), B.min(), C.max(), C.min())
            ABC = np.concatenate([A,B,C], axis=1)

            #use imwshow to display all three images
            ax1=plt.subplot(ncol, nrow, i)
            plt.imshow(ABC)
            plt.axis('off')
            
            slice_index+=1
            del A
            del B
            del C
            del ABC
    
    plt.tight_layout( )
    plt.subplots_adjust(  wspace=0.01, hspace=0.1)
    #outer_grid.tight_layout(fig, pad=0, h_pad=0, w_pad=0) 
    plt.savefig(output_fn, dpi=2000, width=4000)  
    plt.clf()
    return 0


def set_output_image_fn(pet_fn, predict_dir, verbose=1):
    '''
        set output directory for subject and create filename for image slices. 
        output images are saved according to <predict_dir>/<subject name>/...png

        args:
            pet_fn -- filename of pet image on which prection was based
            predict_dir -- output directory for predicted images
            verbose -- print output filename if 2 or greater, 0 by default

        return:
            image_fn -- output filename for slices
    '''
    pet_basename = splitext(basename(pet_fn))[0]
    name=[ f for f in pet_basename.split('_') if 'sub' in f.split('-') ][0]
    
    image_fn = predict_dir +os.sep + pet_basename + '_predict.png'
    
    if verbose >= 2 : print('Saving to:', image_fn) 
    
    return image_fn

def predict_image(i, model, X_all, Y_all, pet_fn, predict_dir, start, end, loss, verbose=1):
    '''
        Slices the input numpy arrays to extract 3d volumes, creates output filename for subject, applies model to X_validate and then saves volume to png.

        args:
            i -- index number of image 
            X_all -- tensor loaded from .npy file with all X_validate stored in it
            Y_all -- tensor loaded from .npy file with all Y_validate stored in it
            pet_fn -- filename of pet image 
            predict_dir -- base directory for predicted images
            samples_per_subject -- number of samples in <X_all> and <Y_all> per subject
        
        return:
            image_fn -- filename of png to which slices were saved
    
    '''
    #get image 3d volume from tensors
    X_validate = X_all[start:end]
    Y_validate = Y_all[start:end]

    #set output filename for png file
    image_fn = set_output_image_fn(pet_fn, predict_dir, verbose)
    image_fn = sub('.png','_'+str(i)+'.png',image_fn)
    #print(image_fn)
    #apply model to X_validate to get predicted values

    X_predict = model.predict(X_validate, batch_size = 1)
    if type(X_predict) != type(np.array([])) : return 1
    
    X_validate = X_validate.reshape(X_validate.shape[0:3])
    print(np.unique(Y_validate))
    exit(0)
    for x in to_categorical(Y_validate,3 ) :
        for y in x :
            for z in y :
                if np.argmax(z) > 0 :
                    print(np.argmax(z))
                    print(z)
    X_predict  = np.argmax(X_predict, axis=3) 
    Y_validate = Y_validate.reshape(Y_validate.shape[0:3])
    print( np.unique(X_predict))
    exit(0)
    #save slices from 3 numpy arrays to <image_fn>
    save_image(X_validate, X_predict,  Y_validate, image_fn)
    del Y_validate
    del X_validate
    del X_predict
    return image_fn


def predict(model_fn, predict_dir, data_dir, images_fn, loss, evaluate=False, category='test', images_to_predict=None, verbose=1 ):
    '''
        Applies model defined in <model_fn> to a set of validate images and saves results to png image
        
        args:
            model_fn -- name of model with stored network weights
            target_dir -- name of target directory where output is saved
            images_to_predict -- images to predict, can either be 'all' or a comma separated string with list of index values of images to save

        return:
            0

    '''
    images = pd.read_csv(images_fn)
    #create new pandas data frame <images> that contains only images marked with category

    images = images[ images.category == category]
    images.index = range(images.shape[0])
    nImages =images.shape[0]
    
    #set which images within images will be predicted
    if images_to_predict == 'all': images_to_predict = range(images.shape[0]) 
    elif type(images_to_predict) == str : images_to_predict =  [int(i) for i in images_to_predict.split(',')]
    #otherwise run prediction for all images
    else: 
        print('No images were specified for prediction.')
        return 0
    
    #check that the model exists and load it
    if exists(model_fn) :
        model = load_model(model_fn)
        if verbose >= 1: print("Model successfully loaded", model)
    else :
        print('Error: could not find', model_fn)
        exit(0)
   
    #load data for prediction
    x_fn=glob(data_dir + os.sep + category + '_x.npy')
    y_fn=glob(data_dir + os.sep + category + '_y.npy')
    if x_fn != [] : x_fn=x_fn[0]
    if y_fn != [] : y_fn=y_fn[0]
    X_all =np.load(x_fn) 
    Y_all = np.load(y_fn)
    if verbose >= 1: print("Data loaded for prediction")

    for i in images_to_predict:
        if i==0 : start_sample = 0
        else :    start_sample = int(images.iloc[0:i,].valid_samples.sum())
        end_sample = int(images.iloc[0:(i+1),].valid_samples.sum())
        pet_fn=images.iloc[i,].pet
        samples_per_subject = images.iloc[i,].valid_samples
        print( images.iloc[i,])
        #print(start_sample, end_sample)
        print(os.path.basename(images.iloc[i,].pet), start_sample, end_sample)
        #print(predict_dir)
        predict_image(i, model, X_all, Y_all, pet_fn, predict_dir, start_sample, end_sample, loss, verbose)


    if verbose >= 1:  print("Prediction completed")
    return 0



if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Process inputs for predict.')
    parser.add_argument('--model', dest='model_fn', type=str,  help='model to use for prediction')
    parser.add_argument('--target', dest='predict_dir', type=str,  help='directory to save predicted images')
    parser.add_argument('--data_dir', dest='data_dir', type=str,  help='data_dir where npy file can be found')
    parser.add_argument('--images', dest='images_fn', type=str,  help='filename with images .csv with information about files')
    parser.add_argument('--category', dest='category', type=str, default='test', help='Image cagetogry: train/validation/test')
    parser.add_argument('--evaluate', dest='evaluate', default=False, action='store_true',  help='Image cagetogry: train/validation/test')
    parser.add_argument('--images_to_predict', dest='images_to_predict', type=str, default='all', help='either 1) \'all\' to predict all images OR a comma separated list of index numbers of images on which to perform prediction (by default perform none). example \'1,4,10\'' )
    parser.add_argument('--verbose', dest='verbose', type=int, default=1, help='verbose level: 0=silent, 1=default')
    args = parser.parse_args()

    predict(model_fn=args.model_fn, predict_dir=args.predict_dir, data_dir=args.data_dir, images_fn=args.images_fn, evaluate=args.evaluate, category=args.category, images_to_predict=args.images_to_predict, verbose=args.verbose )

