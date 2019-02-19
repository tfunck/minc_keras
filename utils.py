import os
import h5py
import numpy as np
from os.path import splitext, basename, exists
from keras import backend as K
global categorical_functions
categorical_functions = ["categorical_crossentropy"]

def from_categorical(cat, img):
    out = np.zeros(img.shape)
    for i, cat0 in  zip(np.unique(img), cat) :
        out = out + cat0 * i
    return(out)
    


def set_model_name(filename, target_dir, ext='.hdf5'):
    '''function to set default model name'''
    return  target_dir+os.sep+splitext(basename(filename))[0]+ext


def safe_h5py_open(filename, mode):
    '''open hdf5 file, exit elegantly on failure'''
    #meera
    # At the moment, this function returns a complicated object "f" that contains
    # the image array somewhere inside of it. 
    # You can modify this function so that it uses nibabel to load in images instead
    # of h5py. In this case, this function should return the actual 3D/4D array.
    # 
    try :
        #meera
        #not sure if this is right, but could try something like :
        #f = nibabel.Load(filename)
        #image_array = np.asarray(f.dataobj)
        #return image_array
        f = h5py.File(filename, mode)
        return f

    except OSError :
        print('Error: Could not open', filename)
        exit(1)


def normalize(A):
    '''performs a simple normalization from 0 to 1 of a numpy array. checks that the image is not a uniform value first

    args
        A -- numpy array

    returns
        numpy array (either A or normalized version of A)
    '''
    std_factor=1
    if np.std(A) > 0 : std_factor=np.std(A)
    A = (A - np.mean(A)) / std_factor

    scale_factor=np.max(A) - A.min()
    if  scale_factor == 0: scale_factor = 1
    A = (A - A.min()) /scale_factor
    return A

def dice_coef(y_true, y_pred):
    y_true_f = np.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

def dice_loss(y_true, y_pred):
    """
    Computes approximate DICE coefficient as a loss by using the negative, computed with the Keras backend. The overlap\
    and total are offset to prevent 0/0, and the values are not rounded in order to keep the gradient information.
    Args:
    :arg y_true: Ground truth
    :arg y_pred: Predicted value for some input
    Returns
    :return: Approximate DICE coefficient.
    """
    ytf = K.flatten(y_true)
    ypf = K.flatten(y_pred)

    overlap = K.sum(ytf*ypf)
    total = K.sum(ytf*ytf) + K.sum(ypf * ypf)
    return -(2*overlap +1e-10) / (total + 1e-10)


def dice_metric(y_true, y_pred):
    """
    Computes DICE coefficient, computed with the Keras backend.
    Args:
    :arg y_true: Ground truth
    :arg y_pred: Predicted value for some input
    Returns
    :return: DICE coefficient
    """
    #ytf = K.round(K.flatten(y_true))
    #ypf = K.round(K.flatten(y_pred))

    #overlap = 2*K.sum(ytf*ypf)
    #total = K.sum(ytf*ytf) + K.sum(ypf * ypf)
    
    #return overlap / total
    return -1 * dice_loss(y_true, y_pred)

