import keras
from keras.models import Sequential, load_model
from keras.layers import Dense, MaxPooling3D, BatchNormalization
from keras.layers.convolutional import Conv1D, Conv2D, Conv3D
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.convolutional import ZeroPadding3D, ZeroPadding2D, ZeroPadding1D
from keras.layers.core import Dropout
def make_model(batch_size):

	model = Sequential()
	model.add(BatchNormalization(batch_input_shape=(batch_size,217,181,1)))
	model.add(Conv2D( 16 , [3,3],  activation='relu',padding='same'))
	model.add(Conv2D(32, (3, 3), activation='relu',padding='same'))
	model.add(Conv2D(64, (3, 3), activation='relu',padding='same'))
	model.add(Conv2D(64, (3, 3), activation='relu',padding='same'))
	model.add(Conv2D(64, (3, 3), activation='relu',padding='same'))
	model.add(Conv2D(64, (3, 3), activation='relu',padding='same'))
	model.add(Conv2D(1, kernel_size=1,  padding='same', activation='sigmoid'))

	return(model)

def compile_and_run(model, X_train, Y_train, X_test, Y_test, batch_size, nb_epoch, lr=5e-3):

	ada = keras.optimizers.Adam(lr)
	model.compile(loss = 'binary_crossentropy', optimizer = ada,metrics=['accuracy'] )
	model.fit(X_train,Y_train, batch_size, validation_data=(X_test, Y_test), epochs = nb_epoch)
	return(model)
