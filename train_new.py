from keras.models import Model
from keras.optimizers import SGD
from keras.layers import Dense, LSTM, Dropout, GRU, Input, RepeatVector, Reshape
import numpy as np
from random import shuffle
from keras.layers.wrappers import Bidirectional, TimeDistributed

inputs=Input(shape=(100,6))
encoder=Bidirectional(GRU(50))(inputs)
encoded_vector=RepeatVector(100)(encoder)
decoder=Bidirectional(GRU(10,return_sequences=True))(encoded_vector)
#decoder=TimeDistributed(Dense(2))(decoder)
decoder=Reshape((2000,))(decoder)
decoder=Dense(200)(decoder)

model=Model(inputs=inputs,outputs=decoder)
optimizer=SGD(0.05,momentum=0.9,nesterov=True,clipnorm=1.)
metrics=['mae']
loss='mse'
model.compile(optimizer=optimizer,metrics=metrics,loss=loss)

train_X=np.load('new_data/train_X.npz')['arr_0']
train_Y=np.load('new_data/train_Y.npz')['arr_0']
a=[]
for i in range(len(train_X)):
	a.append((train_X[i],train_Y[i]))
shuffle(a)
val_X=[]
val_Y=[]
tX=[]
tY=[]
for i in range(int(0.1*len(a))):
	val_X.append(a[i][0])
	val_Y.append(a[i][1])
for i in range(int(0.1*len(a)),len(a)):
	tX.append(a[i][0])
	tY.append(a[i][1])
train_X=np.array(tX)
train_Y=np.array(tY)
val_X=np.array(val_X)
val_Y=np.array(val_Y)
print val_X.shape,val_Y.shape,train_X.shape,train_Y.shape

model.fit(train_X,train_Y.reshape((-1,200)),epochs=10)
print model.evaluate(val_X,val_Y.reshape((-1,200)))
