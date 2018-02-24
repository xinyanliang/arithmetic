import numpy as np
np.random.seed(1337)  # for reproducibility

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam

batch_size = 100
nb_epoch = 250

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
def main():
	test_X = np.load('test_X.npy')

	model = Sequential()
	model.add(Flatten(input_shape=(15,60,2)))
	model.add(Dense(128))
	model.add(Activation('relu'))
	model.add(Dense(128))
	model.add(Activation('relu'))
	model.add(Dense(128))
	model.add(Activation('relu'))
	model.add(Dense(900))
	model.add(Activation('sigmoid'))
	model.load_weights('model.h5')

	adam = Adam(0.001)
	#adagrad = Adagrad(lr=0.01)
	model.compile(loss='mse', optimizer=adam)
	pre_y = model.predict(x=test_X,batch_size=128)
	np.save('pre_y',pre_y)

if __name__ == "__main__":
	main()
