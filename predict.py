import numpy as np
import scipy as sp
import pandas as pd
from os.path import basename, exists, splitext
from os import makedirs
import matplotlib.pyplot as plt
from re import sub
from keras.models import  load_model
from prepare_data import * 

def normalize(A): return (A - np.min(A))/(np.max(A)-np.min(A))

def save_image(X_test, X_predict, Y_test ,output_fn, slices=None, nslices=9 ):
    if slices == None : slices = range(0,  X_test.shape[0], int(X_test.shape[0]/nslices) )
    plt.figure(1 )
    ncol=1
    for s,i in zip(slices, range(0,nslices*ncol,ncol)):
        A=normalize(X_test[s])
        B=normalize(X_predict[s])
        C=normalize(Y_test[s])
        plt.subplot(nslices, ncol, i+1)
        plt.imshow(np.concatenate([A,B,C], axis=1))
        plt.axis('off')
        plt.subplots_adjust(wspace=0,hspace=0)
        
    plt.savefig(output_fn, bbox_inches='tight', dpi=500, pad=0,  pad_inches=0 )  
    return 0


def predict(model_name,  source_directory, target_dir,images, images_to_predict=None ):
    images = images[ images.category == 'test']
    images.index = range(images.shape[0])
    if images_to_predict == 'all': images_to_predict = range(images.shape[0]) 
    elif type(images_to_predict) == str : images_to_predict =  [int(i) for i in images_to_predict.split(',')]
    #otherwise run prediction for all images
    else: 
        print('No images were specified for prediction.')
        return 0
    
    if exists(model_name) :
        model = load_model(model_name)
        print("Model successfully loaded", model)
    
    
    nImages =images.shape[0]
    
    print("Data prepared for prediction")
    X_test_all = np.load(prepare_data.test_x_fn + '.npy')
    Y_test_all = np.load(prepare_data.test_y_fn + '.npy')
    samples_per_subject = prepare_data.samples_per_subject
    #if user gave a comma separated list of image numbers to predict

  

    print(images_to_predict)
    predict_dir = target_dir + 'predict' + os.sep
    
    for i in images_to_predict:
        start = i*samples_per_subject
        end = start + samples_per_subject
        X_test = X_test_all[start:end]
        Y_test = Y_test_all[start:end]
        
        row = images.iloc[i,]
        pet_basename = splitext(basename(row.pet))[0]
        name=[ f for f in pet_basename.split('_') if 'sub' in f.split('-') ][0]
        image_predict_dir=predict_dir+os.sep+name +os.sep
        if not exists(image_predict_dir): makedirs(image_predict_dir)
        image_fn = image_predict_dir + pet_basename + '_predict.png'
        print(image_fn) 
        X_predict = model.predict(X_test, batch_size = prepare_data.batch_size)
        
        X_test = X_test.reshape(X_test.shape[0:3])
        X_predict = X_predict.reshape(X_predict.shape[0:3])
        Y_test = Y_test.reshape(Y_test.shape[0:3])
        save_image(X_test, X_predict,  Y_test, image_fn)
    print("Prediction completed")
   
