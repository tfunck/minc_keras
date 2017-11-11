import keras
from keras.models import Sequential, load_model
from keras.layers import Input, Add, Multiply, Dense, MaxPooling3D, BatchNormalization, Reshape
from keras.layers.convolutional import Conv1D, Conv2D, Conv3D, Convolution2D
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.convolutional import ZeroPadding3D, ZeroPadding2D, ZeroPadding1D, UpSampling2D
from keras.layers.core import Dropout
from keras.utils import to_categorical
from keras.layers import LeakyReLU, MaxPooling2D, concatenate,Conv2DTranspose, merge, ZeroPadding2D
import numpy as np
from predict import save_image
from custom_loss import *
from math import sqrt
def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

'''def make_unet(batch_size, image_dim, images):
    img_rows=image_dim[1]
    img_cols=image_dim[2]

    inputs = Input((img_rows, img_cols, 1))
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool4)
    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv5)

    up6 = concatenate([Conv2DTranspose(256, (2, 2), padding='same')(conv5), conv4], axis=3)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(up6)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv6)

    up7 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv6), conv3], axis=3)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(up7)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv7)

    up8 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv7), conv2], axis=3)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(up8)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv8)

    up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv8), conv1], axis=3)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(up9)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv9)

    conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9)

    model = Model(inputs=[inputs], outputs=[conv10])

    model.compile(optimizer=Adam(lr=1e-5), loss=dice_coef_loss, metrics=[dice_coef])

    return model'''

def make_unet(batch_size, image_dim, images):
    img_rows=image_dim[1]
    img_cols=image_dim[2]
    nMLP=16
    nRshp=int(sqrt(nMLP))
    nUpSm=int(image_dim[0]/nRshp)
    n_labels = len(np.unique(images["onehot"]))
    image = Input(shape=(image_dim[0], image_dim[1],1))
    onehot = Input(shape=(n_labels,)) # replace 5 by n_labels
    
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

    model = keras.models.Model(input=[image, onehot], output=conv10)
    #model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=[jaccard_coef, jaccard_coef_int, 'accuracy'])

    print(model.summary())
    return model





def make_model(batch_size, image_dim, images):
    return make_unet(batch_size, image_dim, images)
    n_labels = len(np.unique(images["onehot"]))

    nMLP=16
    nRshp=int(sqrt(nMLP))
    nUpSm=int(image_dim[0]/nRshp)
    print('nMLP', nMLP,'nRshp', nRshp,'nUpSm', nUpSm)

    lMLPout = Dense(nMLP,activation = 'relu')(onehot)
    lMLPout = Dropout(.2)(lMLPout)
    lMLPout = Dense(nMLP,activation = 'relu')(lMLPout)
    lMLPout = Dropout(.2)(lMLPout)
    lMLPout = Dense(nMLP,activation = 'relu')(lMLPout)
    lMLPout = Dropout(.2)(lMLPout)
    lMLPout = Dense(nMLP,activation = 'relu')(lMLPout)
    lMLPout = Dropout(.2)(lMLPout)
    lMLPout = Reshape((nRshp,nRshp,1))(lMLPout)
    lMLPout= UpSampling2D(size=(nUpSm, nUpSm))(lMLPout)
    cn0=32
    cn1=64
    cn2=128
    cn3=256
    nKer=5
    out = BatchNormalization()(image)
    out = Multiply()([lMLPout, out])

    out = Conv2D( cn0, [nKer,nKer], padding='same')(out)
    out = LeakyReLU(alpha=0.3)(out) 
    out = Conv2D( cn0, [nKer,nKer], padding='same')(out)
    out = LeakyReLU(alpha=0.3)(out) 
    #out = MaxPooling2D(pool_size=(2, 2))(out)


    out = Conv2D( cn1, [nKer,nKer], padding='same')(out)
    out = LeakyReLU(alpha=0.3)(out) 
    out = Conv2D( cn1, [nKer,nKer], padding='same')(out)
    out = LeakyReLU(alpha=0.3)(out) 
    #out = MaxPooling2D(pool_size=(2, 2))(out)
    
    out = Conv2D( cn2 , [nKer,nKer],padding='same')(out)
    out = Conv2D( cn2 , [nKer,nKer],padding='same')(out)

    out = Conv2D( cn2 , [nKer,nKer],padding='same')(out)
    out = LeakyReLU(alpha=0.3)(out)
    out = Conv2D( cn2 , [nKer,nKer],padding='same')(out)
    out = LeakyReLU(alpha=0.3)(out)
    out = Conv2D( cn3 , [nKer,nKer],padding='same')(out)
    out = LeakyReLU(alpha=0.3)(out)
    out = Conv2D( cn3 , [nKer,nKer],padding='same')(out)
    out = LeakyReLU(alpha=0.3)(out)
    out = Dropout(0.5)(out)
    out = Add()([out, image])
    out = Conv2D(1, kernel_size=1,  padding='same', activation='sigmoid')(out)
    model = keras.models.Model(inputs=[image, onehot], outputs=out)

    print(model.summary())
    return(model)


def compile_and_run(model, X_train, train_onehot, Y_train, X_validate, validate_onehot, Y_validate, batch_size, nb_epoch, lr=0.005):
    train_onehot_cat = to_categorical(train_onehot , len(np.unique(train_onehot)) )
    validate_onehot_cat = to_categorical(validate_onehot , len(np.unique(validate_onehot)) )
    
    N=45
    m=10
    batch_start=batch_size*N
    batch_end=batch_size*(N+m)

    ada = keras.optimizers.Adam(0.0001)
    model.compile(loss = 'binary_crossentropy', optimizer = ada,metrics=['accuracy', dice_loss] )


    model.fit([X_train,train_onehot_cat],Y_train, batch_size, validation_data=([X_validate, validate_onehot_cat ], Y_validate), epochs = nb_epoch)
    #model.fit(X_train, Y_train, batch_size, validation_data=(X_validate, Y_validate), epochs = nb_epoch)
    '''for i in range(nb_epoch):
        #model.fit([X_train,train_onehot_cat],Y_train, batch_size, validation_data=([X_validate, validate_onehot_cat ], Y_validate), epochs = nb_epoch)
        model.fit([X_train,train_onehot_cat],Y_train, batch_size, validation_data=([X_validate, validate_onehot_cat ], Y_validate), epochs = 1)
        
        X_predict = model.predict([X_validate[batch_start:batch_end], validate_onehot_cat[batch_start:batch_end]], batch_size = batch_size*m)
        X_validate0 = X_validate.reshape(X_validate.shape[0:3])[batch_start:batch_end]
        X_predict = X_predict.reshape(X_predict.shape[0:3])
        Y_validate0 = Y_validate.reshape(Y_validate.shape[0:3])[batch_start:batch_end]
        
        #save slices from 3 numpy arrays to <image_fn>
        save_image(X_validate0, X_predict,  Y_validate0, "image_"+str(i)+".png", nslices=9)
        print(i)
    '''
    return(model)
