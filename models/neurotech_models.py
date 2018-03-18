import keras
from keras.models import Sequential, load_model
from keras.layers import Input, Add, Multiply, Dense, MaxPooling3D, BatchNormalization, Reshape
from keras.layers.convolutional import Conv1D, Conv2D, Conv3D, Convolution2D
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.convolutional import ZeroPadding3D, ZeroPadding2D, ZeroPadding1D, UpSampling2D
from keras.layers.core import Dropout
from keras.utils import to_categorical
from keras.layers import LeakyReLU, MaxPooling2D, concatenate,Conv2DTranspose, merge, ZeroPadding2D
from keras.activations import relu
from keras.callbacks import History, ModelCheckpoint
import numpy as np
from predict import save_image
#from custom_loss import *
from math import sqrt
from utils import *
import json

def base_model( image_dim,  nK, kernel_size, drop_out, nlabels, activation):
    IN = OUT = Input(shape=(image_dim[1], image_dim[2],1))
    n_layers=int(len(nK))
    kDim=[kernel_size] * n_layers
    for i in range(n_layers):
        OUT = Conv2D( nK[i] , kernel_size=[kDim[i],kDim[i]], activation='relu',padding='same')(OUT)
        OUT = Dropout(drop_out)(OUT)

    OUT = Conv2D(nlabels, kernel_size=1,  padding='same', activation=activation)(OUT)
    model = keras.models.Model(inputs=[IN], outputs=OUT)
    return(model)

def model_0_0( image_dim, nlabels, activation):
    nK=[16,16,32,32,64,64]
    kernel_size = 3 
    drop_out=0
    return base_model( image_dim, nK, kernel_size, drop_out, nlabels, activation)

def model_1_0( image_dim, nlabels, activation):
    '''
    Increase number of layers
    '''
    nK=[16,16,16,32,32,64,64,64]
    kernel_size = 3 
    drop_out=0
    return base_model( image_dim, nK, kernel_size, drop_out, nlabels, activation)



def model_2_0( image_dim, nlabels, activation):
    '''
    Increase kernel size
    '''
    nK=[16,16,32,32,64,64]
    kernel_size = 5 
    drop_out=0
    return base_model( image_dim, nK, kernel_size, drop_out, nlabels, activation)

def model_3_0( image_dim, nlabels, activation):
    '''
    Increase the depth of the layers but keep the total number of parameters
    '''
    nK=[8,8,8,16,16,16,32,32]
    kernel_size = 3
    drop_out=0
    return base_model( image_dim, nK, kernel_size, drop_out, nlabels, activation)

def model_4_0( image_dim, nlabels, activation):
    nK=[16,16,32,32,64,64,64]
    kernel_size = 3 
    drop_out=0.25
    return base_model( image_dim, nK, kernel_size, drop_out, nlabels, activation)

def model_4_1( image_dim, nlabels, activation):
    nK=[16,16,32,32,64,64,64]
    kernel_size = 3 
    drop_out=0.5
    return base_model( image_dim, nK, kernel_size, drop_out, nlabels, activation)
