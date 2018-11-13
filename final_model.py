from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, GRU
from keras.layers.wrappers import Bidirectional, TimeDistributed
from keras.optimizers import SGD
import numpy as np
from auxillary import *

model=Sequential()
model.add(Dense(200,activation='relu',input_shape=(300,)))
model.add(Dense(100,activation='relu'))
model.add(Dense(50,activation='relu'))
model.add(Dense(1,activation='relu'))

optimizer=SGD(0.05,momentum=0.9,nesterov=True,clipnorm=1.)
metrics=['mae']
loss='mse'
model.compile(optimizer=optimizer,metrics=metrics,loss=loss)

concatenated_embeddings=np.load('concatenated_embeddings.npz')['arr_0']
Y=[]
for num in [300000,600000,900000,994656]:
	Y.append(np.load('data/train_Y_'+str(num)+'.npz')['arr_0'])
Y=np.concatenate(Y,axis=0)
model.fit(concatenated_embeddings,Y,epochs=5)
model.save('models/final_model')
# print 'Mae:',error_final_model