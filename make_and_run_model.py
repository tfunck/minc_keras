def make_model(batch_size):

	model = Sequential()
	model.add(BatchNormalization(batch_input_shape=(batch_size,217,181,1)))
	model.add(Conv2D( 16 , [3,3],  activation='relu',padding='same'))
	model.add(Conv2D(32, (3, 3), activation='relu',padding='same'))
	model.add(Conv2D(64, (3, 3), activation='relu',padding='same'))
	model.add(Conv2D(64, (3, 3), activation='relu',padding='same'))
	model.add(Conv2D(1, kernel_size=1,
	                             padding='same',
	                             activation='sigmoid'))

	return(model)

def compile_and_run(model, X, y, batch_size, lr=5e-3):

	ada = keras.optimizers.Adam(lr)
	model.compile(loss = 'binary_crossentropy', optimizer = ada)
	model.fit(X,y, batch_size, epochs = 2)
	return(model)