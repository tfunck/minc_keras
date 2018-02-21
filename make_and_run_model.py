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
from custom_loss import *
from math import sqrt
from utils import *
import json
def make_unet(batch_size, image_dim, images):
    img_rows=image_dim[1]
    img_cols=image_dim[2]
    nMLP=16
    nRshp=int(sqrt(nMLP))
    nUpSm=int(image_dim[0]/nRshp)
    image = Input(shape=(image_dim[0], image_dim[1],1))
    
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

    up6 = merge([UpSampling2D(size=(2, 2))(conv5), conv4], mode='concat', concat_axis=3)
    conv6 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(up6)
    conv6 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(conv6)

    conv6_up = UpSampling2D(size=(2, 2))(conv6)
    conv6_pad = ZeroPadding2D( ((1,0),(1,0)) )(conv6_up)
    up7 = merge([conv6_pad, conv3], mode='concat', concat_axis=3)
    conv7 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(up7)
    conv7 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(conv7)

    up8 = merge([UpSampling2D(size=(2, 2))(conv7), conv2], mode='concat', concat_axis=3)
    conv8 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(up8)
    conv8 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(conv8)

    up9 = merge([UpSampling2D(size=(2, 2))(conv8), conv1], mode='concat', concat_axis=3)
    conv9 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(up9)
    conv9 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(conv9)

    conv10 = Convolution2D(1, 1, 1, activation='sigmoid')(conv9)

    model = keras.models.Model(input=[image], output=conv10)
    #model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=[jaccard_coef, jaccard_coef_int, 'accuracy'])

    print(model.summary())
    return model


def make_dil(batch_size, image_dim, images):
    
    image = Input(shape=(image_dim[0], image_dim[1],1))

    OUT = BatchNormalization()(image)
    #kDim=[3,3,3,3,3,3,3]
    #nK=[21,21,21,21,21,21,21,1]
    #n_dil=[1,1,2,4,8,16,1,1]
    n_dil=[1,2,4,8,16,1]
    n_layers=int(len(n_dil))
    kDim=[5] * n_layers
    nK=[21] * n_layers
    for i in range(n_layers):
        OUT = Conv2D( nK[i] , kernel_size=[kDim[i],kDim[i]], dilation_rate=(n_dil[i],n_dil[i]),activation='relu',padding='same')(OUT)
        OUT = BatchNormalization()(OUT)
        OUT = Dropout(0.25)(OUT)

    OUT = Conv2D(1, kernel_size=1,  padding='same', activation='sigmoid')(OUT)
    model = keras.models.Model(inputs=[image], outputs=OUT)
    return(model)


def make_model(batch_size, image_dim, images):
    model_type='dil'

    if model_type=='unet' : model=make_unet(batch_size, image_dim, images)
    elif model_type=='dil': model=make_dil(batch_size, image_dim, images)
    

    print(model.summary())
    return(model)


def compile_and_run(model, model_name, history_fn, X_train,  Y_train, X_validate, Y_validate, batch_size, nb_epoch, lr=0.005):
    #set checkpoint filename
    checkpoint_fn = splitext(model_name)[0]+"_checkpoint-{epoch:02d}-{val_acc:.2f}.hdf5"
    #set compiler
    ada = keras.optimizers.Adam(0.0001)
    #create history callback
    #history_callback = History()
    #create csv logger callback
    #csv_logger = CSVLogger(splitext(model_name)[0]+ 'training.txt')
    #create checkpoint callback for model
    checkpoint = ModelCheckpoint(checkpoint_fn, monitor='val_loss', verbose=0, save_best_only=True, mode='max')
    #compile the model
    model.compile(loss = 'binary_crossentropy', optimizer=ada,metrics=['accuracy', dice_loss] )
    #fit model
    history = model.fit([X_train],Y_train, batch_size, validation_data=([X_validate], Y_validate), epochs = nb_epoch,callbacks=[ checkpoint])
    #save model   
    model.save(model_name)

    #save history 
    with open(history_fn, 'w+') as fp: json.dump(history.history, fp)

    return([model, history])
