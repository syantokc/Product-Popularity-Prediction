from keras.models import load_model, Model
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, GRU, Merge
from keras.layers.wrappers import Bidirectional, TimeDistributed
from keras.optimizers import SGD
import numpy as np
from auxillary_final import *

U=LSTM

sales_model=Sequential()
sales_model.add(U(100,input_shape=(20,40)))

rating_model=Sequential()
rating_model.add(U(100,input_shape=(20,40)))

titlepos_model=Sequential()
titlepos_model.add(U(100,input_shape=(20,40)))

titleneg_model=Sequential()
titleneg_model.add(U(100,input_shape=(20,40)))

textpos_model=Sequential()
textpos_model.add(U(100,input_shape=(20,40)))

textneg_model=Sequential()
textneg_model.add(U(100,input_shape=(20,40)))

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

model_name='LSTM'
model.load_weights('models/combined_'+model_name)

X=np.load('data/val_X_new_99840.npz')['arr_0']
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
valX=[sX,rX,tipX,tinX,tepX,tenX]
valY=np.load('data/val_Y_new_99840.npz')['arr_0']

# model=load_model('models/combined_'+model_name)
print model.evaluate(valX,valY)
model=Model(inputs=model.input,outputs=model.get_layer('merge_1').output)

embeddings=[]

X,Y,keys=getTrainData(400,'embeddings')
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
embeddings.append(model.predict([sX,rX,tipX,tinX,tepX,tenX]))


embeddings=np.concatenate(embeddings,axis=0)
print embeddings.shape
np.savez_compressed('combined_embeddings_'+model_name,embeddings)
f=open('keys_'+model_name+'.txt','w')
for key in keys:
	f.write(key+'\n')
f.close()