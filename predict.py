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


def save_image(X_test, X_predict, Y_test ,output_fn, slices=None, nslices=25 ):
    '''
        Writes X_test, X_predict, and Y_test to a single png image. Unless specific slices are given, function will write <nslices> evenly spaced slices. 
        
        args:
        X_test -- slice of values input to model
        X_predict -- slice of predicted values based on X_test
        Y_test -- slice of predicted values
        output_fn -- filename of output png file
        slices -- axial slices to save to png file, None by default
        nslices -- number of evenly spaced slices to save to png

        returns: 0
    '''
    #if no slices are defined by user, set slices to evenly sampled slices along entire number of slices in 3d image volume
    if slices == None : slices = range(0,  X_test.shape[0], int(X_test.shape[0]/nslices) )

    
    #set number of rows and columns in output image. currently, sqrt() means that the image will be a square, but this could be changed if a more vertical orientation is prefered
    ncol=int(np.sqrt(nslices))
    nrow=ncol
    
    fig = plt.figure(1 )

    #using gridspec because it seems to give a bit more control over the spacing of the images. define a nrow x ncol grid
    outer_grid = gridspec.GridSpec(nrow, ncol,wspace=0.0, hspace=0.0 )
    
    slice_index=0 #index value for <slices>
    #iterate over columns and rows:
    for col in range(ncol):
        for row in range(nrow) :
            s=slices[slice_index]
            i=col*nrow+row 
            
            #couldn't get inner grid to work properly, so commented out for now. 
            #in theory, should be able to get rid of all white spacing with it
            #inner_grid = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=outer_grid[i], wspace=0.0, hspace=0.0)
            
            #normalize the three input numpy arrays. normalizing them independently is necessary so that they all have the same scale
            A=normalize(X_test[s])
            B=normalize(X_predict[s])
            C=normalize(Y_test[s])
            print(A.max(), B.max(), C.max())
            print(A.mean(), B.mean(), C.mean())
            ABC = np.concatenate([A,B,C], axis=1)

            #use imwshow to display all three images
            plt.subplot(outer_grid[i] )
            plt.imshow(ABC)
            plt.axis('off')
            plt.subplots_adjust(hspace=0.0, wspace=0.0)
            
            slice_index+=1

    outer_grid.tight_layout(fig,  pad=0, h_pad=0, w_pad=0 ) 
    plt.savefig(output_fn, dpi=750)  
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

def predict_image(i, model, X_test_all, Y_test_all, pet_fn, predict_dir,  samples_per_subject, verbose=1):
    '''
        Slices the input numpy arrays to extract 3d volumes, creates output filename for subject, applies model to X_test and then saves volume to png.

        args:
            i -- index number of image 
            X_test_all -- tensor loaded from .npy file with all X_test stored in it
            Y_test_all -- tensor loaded from .npy file with all Y_test stored in it
            pet_fn -- filename of pet image 
            predict_dir -- base directory for predicted images
            samples_per_subject -- number of samples in <X_test_all> and <Y_test_all> per subject
        
        return:
            image_fn -- filename of png to which slices were saved
    
    '''
    #get image 3d volume from tensors
    start = i*samples_per_subject
    end = start + samples_per_subject
    X_test = X_test_all[start:end]
    Y_test = Y_test_all[start:end]
    
    #set output filename for png file
    image_fn = set_output_image_fn(pet_fn, predict_dir, verbose)
   
    #apply model to X_test to get predicted values
    X_predict = model.predict(X_test, batch_size = prepare_data.batch_size)
    
    #reshape all 3 numpy arrays to turn them from (zdim, ydim, xdim, 1) --> (zdim, ydim, xdim)
    X_test = X_test.reshape(X_test.shape[0:3])
    X_predict = X_predict.reshape(X_predict.shape[0:3])
    Y_test = Y_test.reshape(Y_test.shape[0:3])

    #save slices from 3 numpy arrays to <image_fn>
    save_image(X_test, X_predict,  Y_test, image_fn)

    return image_fn


def predict(model_name, target_dir,images, images_to_predict=None, verbose=1 ):
    '''
        Applies model defined in <model_name> to a set of test images and saves results to png image
        
        args:
            model_name -- name of model with stored network weights
            target_dir -- name of target directory where output is saved
            images_to_predict -- images to predict, can either be 'all' or a comma separated string with list of index values of images to save

        return:
            0

    '''
    #create new pandas data frame <test_images> that contains only images marked with category 'test'

    test_images = images[ images.category == 'test']
    test_images.index = range(test_images.shape[0])
    nImages =test_images.shape[0]
    
    #set which images within test_images will be predicted
    if images_to_predict == 'all': images_to_predict = range(test_images.shape[0]) 
    elif type(images_to_predict) == str : images_to_predict =  [int(i) for i in images_to_predict.split(',')]
    #otherwise run prediction for all images
    else: 
        print('No images were specified for prediction.')
        return 0
    

    #check that the model exists and load it
    if exists(model_name) :
        model = load_model(model_name)
        if verbose >= 1: print("Model successfully loaded", model)
    else :
        print('Error: could not find', model_name)
        exit(0)
   
    #load data for prediction
    X_test_all = np.load(prepare_data.test_x_fn + '.npy')
    Y_test_all = np.load(prepare_data.test_y_fn + '.npy')
    samples_per_subject = prepare_data.samples_per_subject
    if verbose >= 1: print("Data loaded for prediction")
  

    predict_dir = target_dir + os.sep + 'predict' + os.sep
    
    for i in images_to_predict:
        pet_fn=test_images.iloc[i,].pet
        predict_image(i, model, X_test_all, Y_test_all, pet_fn, predict_dir,  samples_per_subject, verbose)


    if verbose >= 1:  print("Prediction completed")
    return 0
