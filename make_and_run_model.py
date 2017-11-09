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

def make_model(batch_size, image_dim, images):
    n_labels = len(np.unique(images["onehot"]))
    image = Input(shape=(image_dim[0], image_dim[1],1))
    onehot = Input(shape=(n_labels,)) # replace 5 by n_labels


    lMLP0 = Dense(16,activation = 'relu')
    lMLP1 = Dense(16,activation = 'relu')
    lMLP2 = Dense(image_dim[0]*image_dim[1],activation = 'relu')

    kMLP0 = Dense(16,activation = 'relu')
    kMLP1 = Dense(16,activation = 'relu')
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

    added = Add()([lMLPout, image])
    multiplied = Multiply()([added, kMLPout])

    BN = BatchNormalization()
    C0 = Conv2D( 16 , [3,3],padding='same')
    C1 = Conv2D(32, (3, 3),padding='same')
    C2 = Conv2D(64, (3, 3),padding='same')
    D1 = Dropout(0.5)
    CF = Conv2D(1, kernel_size=1,  padding='same', activation='sigmoid')

    out = BN(multiplied)
    out = C0(out)
    out = LeakyReLU(alpha=0.3)(out)
    out = C1(out)
    out = LeakyReLU(alpha=0.3)(out)
    out = C2(out)
    out = LeakyReLU(alpha=0.3)(out)
    out = D1(out)
    out = CF(out)

    model = keras.models.Model(inputs=[image, onehot], outputs=out)
    print(model.summary())

    return(model)


def compile_and_run(model, X_train, train_onehot, Y_train, X_test, test_onehot, Y_test, batch_size, nb_epoch, lr=0.005):
    print( train_onehot.shape )
    print( test_onehot.shape )
    print( X_train.shape )
    print( X_test.shape )
    print( len(np.unique(train_onehot)) )
    train_onehot_cat = to_categorical(train_onehot , len(np.unique(train_onehot)) )
    test_onehot_cat = to_categorical(test_onehot , len(np.unique(test_onehot)) )
    #train_onehot_cat = train_onehot_cat.astype(float)
    #test_onehot_cat = test_onehot_cat.astype(float)
    #train_onehot_cat = train_onehot_cat.reshape(6,train_onehot.shape[0])
    #test_onehot_cat = test_onehot_cat.reshape(6,test_onehot.shape[0])
    N=45
    m=1
    batch_start=batch_size*N
    batch_end=batch_size*(N+m)

    ada = keras.optimizers.Adam(0.005)
    model.compile(loss = 'binary_crossentropy', optimizer = ada,metrics=['accuracy'] )
    for i in range(nb_epoch):
        #model.fit([X_train,train_onehot_cat],Y_train, batch_size, validation_data=([X_test, test_onehot_cat ], Y_test), epochs = nb_epoch)
        model.fit([X_train,train_onehot_cat],Y_train, batch_size, validation_data=([X_test, test_onehot_cat ], Y_test), epochs = 1)
        
        X_predict = model.predict([X_test[batch_start:batch_end], test_onehot_cat[batch_start:batch_end]], batch_size = batch_size*m)
        X_test0 = X_test.reshape(X_test.shape[0:3])[batch_start:batch_end]
        X_predict = X_predict.reshape(X_predict.shape[0:3])
        Y_test0 = Y_test.reshape(Y_test.shape[0:3])[batch_start:batch_end]
        
        #save slices from 3 numpy arrays to <image_fn>
        save_image(X_test0, X_predict,  Y_test0, "image_"+str(i)+".png", nslices=batch_size*m)
        print(i)

    return(model)
