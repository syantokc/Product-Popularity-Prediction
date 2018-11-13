from keras.models import Sequential,Model
from keras.layers import Dense, Dropout, LSTM, GRU, Input
from keras.layers.wrappers import Bidirectional, TimeDistributed
from keras.layers.merge import concatenate
from keras.optimizers import SGD
import numpy as np
from auxillary import *

n=[300000,600000,900000,994656]

sales_input=Input(shape=(20,40))
sales_diff_x=Input(shape=(1,))
sales_embedding=LSTM(100)(sales_input)
sales_embedding_diff=concatenate([sales_embedding,sales_diff_x])
sales_embedding_diff=Dense(50,activation='relu')(sales_embedding_diff)
sales_output=Dense(1)(sales_embedding_diff)
sales_model=Model(inputs=[sales_input,sales_diff_x],outputs=sales_output)

rating_input=Input(shape=(20,40))
rating_diff_x=Input(shape=(1,))
rating_embedding=LSTM(100)(rating_input)
rating_embedding_diff=concatenate([rating_embedding,rating_diff_x])
rating_embedding_diff=Dense(50,activation='relu')(rating_embedding_diff)
rating_output=Dense(1)(rating_embedding_diff)
rating_model=Model(inputs=[rating_input,rating_diff_x],outputs=rating_output)

sentiment_input=Input(shape=(20,40))
sentiment_diff_x=Input(shape=(1,))
sentiment_embedding=LSTM(100)(sentiment_input)
sentiment_embedding_diff=concatenate([sentiment_embedding,sentiment_diff_x])
sentiment_embedding_diff=Dense(50,activation='relu')(sentiment_embedding_diff)
sentiment_output=Dense(1)(sentiment_embedding_diff)
sentiment_model=Model(inputs=[sentiment_input,sentiment_diff_x],outputs=sentiment_output)

optimizer=SGD(0.05,momentum=0.9,nesterov=True,clipnorm=1.)
metrics=['mae']
loss='mse'
sales_model.compile(optimizer=optimizer,metrics=metrics,loss=loss)
rating_model.compile(optimizer=optimizer,metrics=metrics,loss=loss)
sentiment_model.compile(optimizer=optimizer,metrics=metrics,loss=loss)

for i in range(20):
	print 'Epoch no:',i+1
	for num in n:
		print 'Loading',num,
		X=np.load('data/train_X_'+str(num)+'.npz')['arr_0']
		Y=np.load('data/train_Y_'+str(num)+'.npz')['arr_0']
		diff=np.load('data/train_xdifference_X'+str(num)+'.npz')['arr_0']
		print 'Done'
		sX=X[0].reshape((-1,2,400,1))
		sX=np.concatenate([sX[:,0,:,:],sX[:,1,:,:]],axis=-1).reshape((-1,20,40))
		sales_model.fit([sX,diff[:,0].reshape((-1,1))],Y,epochs=1)
		sX=X[1].reshape((-1,2,400,1))
		sX=np.concatenate([sX[:,0,:,:],sX[:,1,:,:]],axis=-1).reshape((-1,20,40))
		rating_model.fit([sX,diff[:,1].reshape((-1,1))],Y,epochs=1)
		sX=X[2].reshape((-1,2,400,1))
		sX=np.concatenate([sX[:,0,:,:],sX[:,1,:,:]],axis=-1).reshape((-1,20,40))
		sentiment_model.fit([sX,diff[:,2].reshape((-1,1))],Y,epochs=1)

y_true=[]
y_pred_ratings=[]
y_pred_sentiment=[]
for num in n:
	X=np.load('data/train_X_'+str(num)+'.npz')['arr_0']
	Y=np.load('data/train_Y_'+str(num)+'.npz')['arr_0']
	diff=np.load('data/train_xdifference_X'+str(num)+'.npz')['arr_0']
	y_true.append(Y.flatten())
	sX=X[0].reshape((-1,2,400,1))
	sX=np.concatenate([sX[:,0,:,:],sX[:,1,:,:]],axis=-1).reshape((-1,20,40))
	y_pred_ratings.append(rating_model.predict([sX,diff[:,1].reshape((-1,1))]).flatten())
	sX=X[1].reshape((-1,2,400,1))
	sX=np.concatenate([sX[:,0,:,:],sX[:,1,:,:]],axis=-1).reshape((-1,20,40))
	y_pred_sentiment.append(sentiment_model.predict([sX,diff[:,2].reshape((-1,1))]).flatten())
y_true=np.concatenate(y_true,axis=-1)
y_pred_ratings=np.concatenate(y_pred_ratings,axis=-1)
y_pred_sentiment=np.concatenate(y_pred_sentiment,axis=-1)
sales_model.save('models/sales_model_diff_added')
rating_model.save('models/rating_model_diff_added')
sentiment_model.save('models/sentiment_model_diff_added')
print 'Saved models'
print 'Ratings model error:',mae(y_true,y_pred_ratings)
print 'Sentiment model error:',mae(y_true,y_pred_sentiment)
# print 'Sales model error:',error_sales_diffx_added(sales_model,400,(-1,20,40))