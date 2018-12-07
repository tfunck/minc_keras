import keras
from keras.models import Sequential, load_model
from keras.layers import Input, Add, Multiply, Dense, MaxPooling3D, BatchNormalization, Reshape
from keras.layers.convolutional import Conv1D, Conv2D, Conv3D, Convolution2D
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.convolutional import ZeroPadding3D, ZeroPadding2D, ZeroPadding1D, UpSampling2D
from keras.layers.core import Dropout
from keras.utils import to_categorical
from keras.layers import LeakyReLU, MaxPooling2D, concatenate,Conv2DTranspose, Concatenate, ZeroPadding2D
from keras.activations import relu
from keras.callbacks import History, ModelCheckpoint
import numpy as np
from predict import save_image
#from custom_loss import *
#from models.neurotech_models import *
from math import sqrt
from utils import *
import json

def make_unet( image_dim, nlabels, activation_hidden, activation_output, verbose=0):
    img_rows=image_dim[1]
    img_cols=image_dim[2]
    nMLP=16
    nRshp=int(sqrt(nMLP))
    nUpSm=int(image_dim[0]/nRshp)
    image = Input(shape=(image_dim[1], image_dim[2],1))
    n_downsample=4
    #if x != 0 or y != 0 :
    #    print("Error: image must have dimensions that can be divided by "+"2^"+str(n_downsample)+" but has dimensions "+str(image_dim[1])+","+str(image_dim[1]) )
    #    exit(1)

    BN1 = BatchNormalization()(image)

    conv1 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(BN1)
    conv1 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
 
    conv2 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(pool1)
    conv2 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(pool2)
    conv3 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(pool3)
    conv4 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
 
    conv5 = Convolution2D(512, 3, 3, activation='relu', border_mode='same')(pool4)
    conv5 = Convolution2D(512, 3, 3, activation='relu', border_mode='same')(conv5)
    
    up5 = UpSampling2D(size=(2, 2))(conv5)
    #up5 = Conv2DTranspose( filters=512, kernel_size=(3,3), strides=(2, 2), padding='same')(conv5)
    conc5 = Concatenate(axis=3)([up5, conv4])
    conv6 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(conc5)
    conv6 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(conv6)

    up6 = UpSampling2D(size=(2, 2))(conv6)
    #up6 = Conv2DTranspose( filters=512, kernel_size=(3,3), strides=(2, 2), padding='same')(conv6)
    conc6 = Concatenate(axis=3)([up6, conv3])
    conv7 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(up6)
    conv7 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(conv7)

    up7 = UpSampling2D(size=(2, 2))(conv7)
    #up7 = Conv2DTranspose( filters=512, kernel_size=(3,3), strides=(2, 2), padding='same')(conv7)
    conc7 = Concatenate(axis=3)([up7, conv2])
    conv8 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(conc7) #(up8)
    conv8 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(conv8)

    up8 = Conv2DTranspose( filters=512, kernel_size=(3,3), strides=(2, 2), padding='same')(conv8)
    conc8 = Concatenate(axis=3)([up8, conv1])
    conv9 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(conc8) #(up9)
    conv9 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(conv9)

    conv10 = Convolution2D(nlabels, 1, 1, activation=activation_output)(conv9)

    model = keras.models.Model(input=[image], output=conv10)
    
    if verbose > 0 :
        print(model.summary())
    return model


def make_dil( image_dim):
    
    image = Input(shape=(image_dim[1], image_dim[2],1))

    OUT = BatchNormalization()(image)
    #kDim=[3,3,3,3,3,3,3]
    #nK=[21,21,21,21,21,22,21,1]
    n_dil=[1,2,4,8,16,1]
    #n_dil=[1,1,1,2,2,4,4,8,16,1,1]
    n_layers=int(len(n_dil))
    kDim=[6] * n_layers
    nK=[26] * n_layers
    for i in range(n_layers):
        OUT = Conv2D( nK[i] , kernel_size=[kDim[i],kDim[i]], dilation_rate=(n_dil[i],n_dil[i]),activation='relu',padding='same')(OUT)
        OUT = BatchNormalization()(OUT)
        OUT = Dropout(0.25)(OUT)

    OUT = Conv2D(1, kernel_size=1,  padding='same', activation='sigmoid')(OUT)
    model = keras.models.Model(inputs=[image], outputs=OUT)
    return(model)


def base_model( image_dim,  nlabels, nK, n_dil, kernel_size, drop_out, activation_hidden, activation_output, verbose=1):
    print("N Labels:", nlabels)
    print("Drop out:",drop_out)
    print("Number of Dilations:", n_dil)
    print("Activation hidden:", activation_hidden)
    print("Activation output:", activation_output)
    nK=[int(i) for i in nK.split(",") ]
    if n_dil == None :
        n_dil=[1] * len(nK)
    else: 
        n_dil=[int(i) for i in n_dil.split(",") ]
    
    IN = CONV = Input(shape=(image_dim[1], image_dim[2],1))
    n_layers=int(len(nK))
    kDim=[kernel_size] * n_layers

    for i in range(n_layers):
        print("Layer:", i, nK[i], kDim[i], n_dil[i])
        CONV = Conv2D(nK[i], kernel_size=[kDim[i],kDim[i]],dilation_rate=(n_dil[i],n_dil[i]), activation=activation_hidden,padding='same')(CONV)
        CONV = Dropout(drop_out)(CONV)

    OUT = Conv2D(nlabels,  kernel_size=[1,1], activation=activation_output,  padding='same')(CONV)
    model = keras.models.Model(inputs=[IN], outputs=OUT)
    if verbose > 0 :
        print(model.summary())
    
    return(model)


def make_model( image_dim, nlabels,nK, n_dil, kernel_size, drop_out, model_type='model_0_0', activation_hidden="relu", activation_output="sigmoid", verbose=0):
    if model_type=='unet' : 
        model=make_unet( image_dim, nlabels, activation_hidden, activation_output)
    #elif model_type=='dil': model=make_dil( image_dim, nlabels, activation_hidden, activation_output)
    #elif model_type=='model_0_0': model=model_0_0( image_dim, nlabels, nK, kernel_size, drop_out, activation_hidden, activation_output)
    #elif model_type=='model_1_0': model=model_1_0( image_dim, nlabels, nK, kernel_size, drop_out, activation_hidden, activation_output)
    #elif model_type=='model_1_1': model=model_1_1( image_dim, nlabels, nK, kernel_size, drop_out, activation_hidden, activation_output)
    #elif model_type=='model_2_0': model=model_2_0( image_dim, nlabels, nK, kernel_size, drop_out, activation_hidden, activation_output)
    #elif model_type=='model_2_1': model=model_2_1( image_dim, nlabels, nK, kernel_size, drop_out, activation_hidden, activation_output)
    #elif model_type=='model_3_0': model=model_3_0( image_dim, nlabels, nK, kernel_size, drop_out, activation_hidden, activation_output)
    #elif model_type=='model_3_1': model=model_3_1( image_dim, nlabels, nK, kernel_size, drop_out, activation_hidden, activation_output)
    #elif model_type=='model_4_0': model=model_4_0( image_dim, nlabels, nK, kernel_size, drop_out, activation_hidden, activation_output)
    #elif model_type=='model_4_1': model=model_4_1( image_dim, nlabels, nK, kernel_size, drop_out, activation_hidden, activation_output)
    else :             
        model=base_model( image_dim,  nlabels, nK, n_dil, kernel_size, drop_out, activation_hidden, activation_output)
    
    return(model)

def compile_and_run(model, model_name, history_fn, X_train,  Y_train, X_validate, Y_validate,  nb_epoch, nlabels, metric="categorical_accuracy", loss='categorical_crossentropy', lr=0.005, verbose=0):
    #set compiler
    ada = keras.optimizers.Adam(0.0001)
    #set checkpoint filename
    checkpoint_fn = splitext(model_name)[0]+"_checkpoint-{epoch:02d}-{val_loss:.2f}.hdf5"
    #create checkpoint callback for model
    checkpoint = ModelCheckpoint(checkpoint_fn, monitor='val_loss', verbose=0, save_best_only=True, mode='max')
    #compile the model
    model.compile(loss = loss, optimizer=ada,metrics=[metric] )
    #fit model

    X_train = X_train
    X_validate = X_validate
    if loss in categorical_functions : 
        Y_train = to_categorical(Y_train, num_classes=nlabels)
        Y_validate = to_categorical(Y_validate, num_classes=nlabels)

    history = model.fit([X_train],Y_train,  validation_data=([X_validate], Y_validate), epochs = nb_epoch,callbacks=[ checkpoint])
    #save model   
    model.save(model_name)

    with open(history_fn, 'w+') as fp: json.dump(history.history, fp)

    return([model, history])
