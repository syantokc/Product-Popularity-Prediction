from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, GRU
from keras.layers.wrappers import Bidirectional, TimeDistributed
from keras.optimizers import SGD
import numpy as np
from auxillary import *

n=[300000,600000,900000,994656]

sales_model=Sequential()
sales_model.add(LSTM(100,input_shape=(20,40)))
sales_model.add(Dense(50,activation='relu'))
sales_model.add(Dense(1))

rating_model=Sequential()
rating_model.add(LSTM(100,input_shape=(20,40)))
rating_model.add(Dense(50,activation='relu'))
rating_model.add(Dense(1))

sentiment_model=Sequential()
sentiment_model.add(LSTM(100,input_shape=(20,40)))
sentiment_model.add(Dense(50,activation='relu'))
sentiment_model.add(Dense(1))

optimizer=SGD(0.05,momentum=0.9,nesterov=True,clipnorm=1.)
metrics=['mae']
loss='mae'
sales_model.compile(optimizer=optimizer,metrics=metrics,loss=loss)
rating_model.compile(optimizer=optimizer,metrics=metrics,loss=loss)
sentiment_model.compile(optimizer=optimizer,metrics=metrics,loss=loss)

for i in range(20):
	print 'Epoch no:',i+1
	for num in n:
		print 'Loading',num,
		X=np.load('data/train_X_'+str(num)+'.npz')['arr_0']
		Y=np.load('data/train_Y_'+str(num)+'.npz')['arr_0']
		print 'Done'
		sX=X[0].reshape((-1,2,400,1))
		sX=np.concatenate([sX[:,0,:,:],sX[:,1,:,:]],axis=-1).reshape((-1,20,40))
		sales_model.fit(sX,Y,epochs=1)
		sX=X[1].reshape((-1,2,400,1))
		sX=np.concatenate([sX[:,0,:,:],sX[:,1,:,:]],axis=-1).reshape((-1,20,40))
		rating_model.fit(sX,Y,epochs=1)
		sX=X[2].reshape((-1,2,400,1))
		sX=np.concatenate([sX[:,0,:,:],sX[:,1,:,:]],axis=-1).reshape((-1,20,40))
		sentiment_model.fit(sX,Y,epochs=1)

y_true=[]
y_pred_ratings=[]
y_pred_sentiment=[]
for num in n:
	X=np.load('data/train_X_'+str(num)+'.npz')['arr_0']
	Y=np.load('data/train_Y_'+str(num)+'.npz')['arr_0']
	y_true.append(Y.flatten())
	sX=X[0].reshape((-1,2,400,1))
	sX=np.concatenate([sX[:,0,:,:],sX[:,1,:,:]],axis=-1).reshape((-1,20,40))
	y_pred_ratings.append(rating_model.predict(sX).flatten())
	sX=X[1].reshape((-1,2,400,1))
	sX=np.concatenate([sX[:,0,:,:],sX[:,1,:,:]],axis=-1).reshape((-1,20,40))
	y_pred_sentiment.append(sentiment_model.predict(sX).flatten())
y_true=np.concatenate(y_true,axis=-1)
y_pred_ratings=np.concatenate(y_pred_ratings,axis=-1)
y_pred_sentiment=np.concatenate(y_pred_sentiment,axis=-1)
sales_model.save('models/sales_model')
rating_model.save('models/rating_model')
sentiment_model.save('models/sentiment_model')
print 'Saved models'
print 'Ratings model error:',mae(y_true,y_pred_ratings)
print 'Sentiment model error:',mae(y_true,y_pred_sentiment)
print 'Sales model error:',error_sales(sales_model,400,(-1,20,40))