from auxillary import *
from random import shuffle
from keras.layers import Dense, LSTM, GRU, Input, Dropout
from keras.models import Model
from keras.optimizers import SGD


embedding_size=50
X,Y=getStandaloneTrainingData(250)
trainX=[]
valX=[]
testX=[]
trainY=[]
valY=[]
testY=[]

idx=[i for i in range(len(X))]
shuffle(idx)
train_length=(7*len(X))/10
val_length=(len(X))/10
test_length=len(X) - train_length - val_length
for t_idx in idx[:train_length]:
	trainX.append(X[t_idx])
	trainY.append(Y[t_idx])
for v_idx in idx[train_length:train_length+val_length]:
	valX.append(X[v_idx])
	valY.append(Y[v_idx])
for t_idx in idx[train_length+val_length:]:
	testX.append(X[t_idx])
	testY.append(Y[t_idx])
trainX=np.array(trainX)
trainY=np.array(trainY)
valX=np.array(valX)
valY=np.array(valY)
testX=np.array(testX)
testY=np.array(testY)

print 'Running for standalone product sales prediction...'
print 'Window size: 250'
print 'Train:',len(trainX)
print 'Val:',len(valX)
print 'Test:',len(testX)
print 'Iter:',20
print 'Loss: MAE'
print 'Embedding size:',embedding_size

def getDenseModel():
	inputs=Input(shape=(250,))
	dense_1=Dense(embedding_size,activation='relu')(inputs)
	dense_2=Dense(1,activation='linear')(dense_1)

	model=Model(inputs=inputs,outputs=dense_2)

	optimizer=SGD(0.05,momentum=0.9,nesterov=True,clipnorm=1.)
	metrics=['mae']
	loss='mae'
	model.compile(optimizer=optimizer,metrics=metrics,loss=loss)

	return model

def getSeqModel(U):
	inputs=Input(shape=(250/embedding_size,embedding_size))
	u=U(embedding_size,activation='relu')(inputs)
	dense_2=Dense(1,activation='linear')(u)

	model=Model(inputs=inputs,outputs=dense_2)

	optimizer=SGD(0.05,momentum=0.9,nesterov=True,clipnorm=1.)
	metrics=['mae']
	loss='mae'
	model.compile(optimizer=optimizer,metrics=metrics,loss=loss)

	return model

models=[('dense',(-1,250),getDenseModel()),('lstm',(-1,250/embedding_size,embedding_size),getSeqModel(LSTM)),('gru',(-1,250/embedding_size,embedding_size),getSeqModel(GRU))]


for (name,reshape_param,model) in models:
	print 'Training',name,'model'
	model_trainX=trainX.reshape(reshape_param)
	model_valX=valX.reshape(reshape_param)
	model_testX=testX.reshape(reshape_param)
	for i in range(1,21):
		print 'Iteration',i
		model.fit(model_trainX,trainY,epochs=1,batch_size=32)
		print 'Val result:',model.evaluate(model_valX,valY)
		print 'Test result:',model.evaluate(model_testX,testY)
