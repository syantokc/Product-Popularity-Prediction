from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, GRU, Merge
from keras.layers.wrappers import Bidirectional, TimeDistributed
from keras.optimizers import SGD
import numpy as np
# from auxillary import *

n=[200000,400000,600000,800000,894816]

U=LSTM

sales_model=Sequential()
sales_model.add(Bidirectional(U(100,return_sequences=True),input_shape=(20,40)))
sales_model.add(U(100))

rating_model=Sequential()
rating_model.add(Bidirectional(U(100,return_sequences=True),input_shape=(20,40)))
rating_model.add(U(100))

titlepos_model=Sequential()
titlepos_model.add(Bidirectional(U(100,return_sequences=True),input_shape=(20,40)))
titlepos_model.add(U(100))

titleneg_model=Sequential()
titleneg_model.add(Bidirectional(U(100,return_sequences=True),input_shape=(20,40)))
titleneg_model.add(U(100))

textpos_model=Sequential()
textpos_model.add(Bidirectional(U(100,return_sequences=True),input_shape=(20,40)))
textpos_model.add(U(100))

textneg_model=Sequential()
textneg_model.add(Bidirectional(U(100,return_sequences=True),input_shape=(20,40)))
textneg_model.add(U(100))

model=Sequential()
model.add(Merge([sales_model,rating_model,titlepos_model,titleneg_model,textpos_model,textneg_model],mode='concat'))
model.add(Dense(150,activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(50,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1))

optimizer=SGD(0.05,momentum=0.9,nesterov=True,clipnorm=1.)
metrics=['mae']
loss='mae'
model.compile(optimizer=optimizer,metrics=metrics,loss=loss)

# X=np.load('data/val_X_new_99840.npz')['arr_0']
# sX=X[0].reshape((-1,2,400,1))
# sX=np.concatenate([sX[:,0,:,:],sX[:,1,:,:]],axis=-1).reshape((-1,20,40))
# rX=X[1].reshape((-1,2,400,1))
# rX=np.concatenate([rX[:,0,:,:],rX[:,1,:,:]],axis=-1).reshape((-1,20,40))
# tipX=X[2].reshape((-1,2,400,1))
# tipX=np.concatenate([tipX[:,0,:,:],tipX[:,1,:,:]],axis=-1).reshape((-1,20,40))
# tinX=X[3].reshape((-1,2,400,1))
# tinX=np.concatenate([tinX[:,0,:,:],tinX[:,1,:,:]],axis=-1).reshape((-1,20,40))
# tepX=X[4].reshape((-1,2,400,1))
# tepX=np.concatenate([tepX[:,0,:,:],tepX[:,1,:,:]],axis=-1).reshape((-1,20,40))
# tenX=X[5].reshape((-1,2,400,1))
# tenX=np.concatenate([tenX[:,0,:,:],tenX[:,1,:,:]],axis=-1).reshape((-1,20,40))
# valX=[sX,rX,tipX,tinX,tepX,tenX]
# valY=np.load('data/val_Y_new_99840.npz')['arr_0']

print 'Loaded val data'

min_mae=100

for i in range(20):
	print 'Epoch no:',i+1
	for num in n:
		print 'Loading',num,
		X=np.load('data/train_X_new_'+str(num)+'.npz')['arr_0']
		Y=np.load('data/train_Y_new_'+str(num)+'.npz')['arr_0']
		print 'Done'
		sX=np.array(X[0]).reshape((-1,2,400,1))
		sX=np.concatenate([sX[:,0,:,:],sX[:,1,:,:]],axis=-1).reshape((-1,20,40))
		rX=np.array(X[1]).reshape((-1,2,400,1))
		rX=np.concatenate([rX[:,0,:,:],rX[:,1,:,:]],axis=-1).reshape((-1,20,40))
		tipX=np.array(X[2]).reshape((-1,2,400,1))
		tipX=np.concatenate([tipX[:,0,:,:],tipX[:,1,:,:]],axis=-1).reshape((-1,20,40))
		tinX=np.array(X[3]).reshape((-1,2,400,1))
		tinX=np.concatenate([tinX[:,0,:,:],tinX[:,1,:,:]],axis=-1).reshape((-1,20,40))
		tepX=np.array(X[4]).reshape((-1,2,400,1))
		tepX=np.concatenate([tepX[:,0,:,:],tepX[:,1,:,:]],axis=-1).reshape((-1,20,40))
		tenX=np.array(X[5]).reshape((-1,2,400,1))
		tenX=np.concatenate([tenX[:,0,:,:],tenX[:,1,:,:]],axis=-1).reshape((-1,20,40))
		model.fit([sX,rX,tipX,tinX,tepX,tenX],Y,epochs=1)
		mae=model.evaluate(valX,valY)[0]
		print 'Mae:',mae
		if mae<min_mae:
			min_mae=mae
			model.save('models/combined_BLSTM_LSTM')
