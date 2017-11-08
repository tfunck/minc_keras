import keras
from keras.models import Sequential, load_model
from keras.layers import Input, Add, Multiply, Dense, MaxPooling3D, BatchNormalization
from keras.layers.convolutional import Conv1D, Conv2D, Conv3D
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.convolutional import ZeroPadding3D, ZeroPadding2D, ZeroPadding1D
from keras.layers.core import Dropout
from keras.utils import to_categorical
from keras.layers import LeakyReLU
import numpy as np
def make_model(batch_size, image_dim, images):

    n_labels = len(np.unique(images.radiotracer))
    image = Input(shape=(image_dim[0], image_dim[1], 1))
    onehot = Input(shape=(n_labels,)) 


    lMLP0 = Dense(32,activation = 'relu')
    lMLP1 = Dense(32,activation = 'relu')
    lMLP2 = Dense(28*28,activation = 'relu')

    kMLP0 = Dense(32,activation = 'relu')
    kMLP1 = Dense(32,activation = 'relu')
    kMLP2 = Dense(28*28,activation = 'relu')

    lMLPout = lMLP0(onehot)
    lMLPout = lMLP1(lMLPout)
    lMLPout = Dropout(.2)(lMLPout)
    lMLPout = lMLP2(lMLPout)

    kMLPout = kMLP0(onehot)
    kMLPout = kMLP1(kMLPout)
    kMLPout = Dropout(.2)(kMLPout)
    kMLPout = kMLP2(kMLPout)

    added = Add()([lMLPout, image])
    multiplied = Multiply()([added, kMLPout])

    C0 = Conv2D( 16 , [3,3],padding='same')
    C1 = Conv2D(32, (3, 3),padding='same')
    C2 = Conv2D(64, (3, 3),padding='same')
    CF = Conv2D(1, kernel_size=1,  padding='same', activation='sigmoid')

    out = C0(multiplied)
    out = LeakyReLU(alpha=0.3)(out)
    out = C1(out)
    out = LeakyReLU(alpha=0.3)(out)
    out = C2(out)
    out = LeakyReLU(alpha=0.3)(out)
    out = CF(out)

    model = keras.models.Model(inputs=[image, onehot], outputs=out)
    print(model.summary())

    return(model)

def compile_and_run(model, X_train, train_onehot, Y_train, X_test, test_onehot, Y_test, batch_size, nb_epoch, lr=0.005):
    train_onehot_cat = to_categorical(train_onehot , len(np.unique(train_onehot)) )
    test_onehot_cat = to_categorical(test_onehot , len(np.unique(test_onehot)) )
    #train_onehot_cat = train_onehot_cat.astype(float)
    #test_onehot_cat = test_onehot_cat.astype(float)
    #train_onehot_cat = train_onehot_cat.reshape(6,train_onehot.shape[0])
    #test_onehot_cat = test_onehot_cat.reshape(6,test_onehot.shape[0])
    ada = keras.optimizers.Adam(0.005)
    model.compile(loss = 'binary_crossentropy', optimizer = ada,metrics=['accuracy'] )
    model.fit([X_train,train_onehot_cat],Y_train, batch_size, validation_data=([X_test, test_onehot_cat ], Y_test), epochs = nb_epoch)
    return(model)
