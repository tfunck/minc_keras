import keras
from keras.models import Sequential, load_model
from keras.layers import Input, Add, Multiply, Dense, MaxPooling3D, BatchNormalization, Reshape
from keras.layers.convolutional import Conv1D, Conv2D, Conv3D
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.convolutional import ZeroPadding3D, ZeroPadding2D, ZeroPadding1D
from keras.layers.core import Dropout
from keras.utils import to_categorical
from keras.layers import LeakyReLU
import numpy as np
from predict import save_image
from custom_loss import *

def make_model(batch_size, image_dim, images):
    n_labels = len(np.unique(images["onehot"]))
    image = Input(shape=(image_dim[0], image_dim[1],1))
    onehot = Input(shape=(n_labels,)) # replace 5 by n_labels
    nMLP=16
    lMLP0 = Dense(nMLP,activation = 'relu')
    lMLP1 = Dense(nMLP,activation = 'relu')
    lMLP2 = Dense(image_dim[0]*image_dim[1],activation = 'relu')
    kMLP0 = Dense(nMLP,activation = 'relu')
    kMLP1 = Dense(nMLP,activation = 'relu')
    kMLP2 = Dense(image_dim[0]*image_dim[1],activation = 'relu')
    lMLPout = lMLP0(onehot)
    lMLPout = lMLP1(lMLPout)
    lMLPout = Dropout(.2)(lMLPout)
    lMLPout = lMLP2(lMLPout)
    lMLPout = Reshape((image_dim[0],image_dim[1],1))(lMLPout)
    kMLPout = kMLP0(onehot)
    kMLPout = kMLP1(kMLPout)
    kMLPout = Dropout(.2)(kMLPout)
    kMLPout = kMLP2(kMLPout)
    kMLPout = Reshape((image_dim[0],image_dim[1],1))(kMLPout)
    '''
    added = Add()([lMLPout, image])
    multiplied = Multiply()([added, kMLPout])''
    '''



    BN1 = BatchNormalization()(image)
    out = Add()([lMLPout, BN1])
    out = Multiply()([kMLPout,out])
    out = Conv2D( 16 , [3,3],padding='same')(out)
    out = LeakyReLU(alpha=0.3)(out)
    out = Conv2D( 32, (3, 3),padding='same')(out)
    out = LeakyReLU(alpha=0.3)(out)
    out = Conv2D( 64, (3, 3),padding='same')(out)
    out = LeakyReLU(alpha=0.3)(out)
    out = Dropout(0.5)(out)
    out = Add()([out, image])
    out = Conv2D(1, kernel_size=1,  padding='same', activation='sigmoid')(out)

    #out = BN(multiplied)
    #out = BN(image)
    #out = C0(out)
    #out = LeakyReLU(alpha=0.3)(out)
    #out = C1(out)
    #out = LeakyReLU(alpha=0.3)(out)
    #out = C2(out)
    #out = LeakyReLU(alpha=0.3)(out)
    #out = C3(out)
    #out = LeakyReLU(alpha=0.3)(out)
    #out = D2(out)
    #out = A2([out,image])
    #out = CF(out)

    model = keras.models.Model(inputs=[image, onehot], outputs=out)
    print(model.summary())

    return(model)


def compile_and_run(model, X_train, train_onehot, Y_train, X_validate, validate_onehot, Y_validate, batch_size, nb_epoch, lr=0.005):
    print( train_onehot.shape )
    print( validate_onehot.shape )
    print( X_train.shape )
    print( X_validate.shape )
    print( len(np.unique(train_onehot)) )
    train_onehot_cat = to_categorical(train_onehot , len(np.unique(train_onehot)) )
    validate_onehot_cat = to_categorical(validate_onehot , len(np.unique(validate_onehot)) )
    
    N=45
    m=10
    batch_start=batch_size*N
    batch_end=batch_size*(N+m)

    ada = keras.optimizers.Adam(0.005)
    model.compile(loss = 'binary_crossentropy', optimizer = ada,metrics=['accuracy', dice_loss] )


    model.fit([X_train,train_onehot_cat],Y_train, batch_size, validation_data=([X_validate, validate_onehot_cat ], Y_validate), epochs = nb_epoch)
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
