from keras.models import load_model, Model
import numpy as np

sales_model=load_model('models/sales_model')
# sales_model.summary()
sales_embedding_model=Model(inputs=sales_model.input,outputs=sales_model.get_layer('lstm_1').output)
rating_model=load_model('models/rating_model')
# rating_model.summary()
rating_embedding_model=Model(inputs=rating_model.input,outputs=rating_model.get_layer('lstm_2').output)
sentiment_model=load_model('models/sentiment_model')
# sentiment_model.summary()
sentiment_embedding_model=Model(inputs=sentiment_model.input,outputs=sentiment_model.get_layer('lstm_3').output)

concatenated_embeddings=[]

for num in [300000,600000,900000,994656]:
	print 'Loading',num,
	X=np.load('data/train_X_'+str(num)+'.npz')['arr_0']
	Y=np.load('data/train_Y_'+str(num)+'.npz')['arr_0']
	print 'Done'
	sX=X[0].reshape((-1,2,400,1))
	sX=np.concatenate([sX[:,0,:,:],sX[:,1,:,:]],axis=-1).reshape((-1,20,40))
	sales_model_embeddings=sales_embedding_model.predict(sX)
	sX=X[1].reshape((-1,2,400,1))
	sX=np.concatenate([sX[:,0,:,:],sX[:,1,:,:]],axis=-1).reshape((-1,20,40))
	rating_model_embeddings=rating_embedding_model.predict(sX)
	sX=X[2].reshape((-1,2,400,1))
	sX=np.concatenate([sX[:,0,:,:],sX[:,1,:,:]],axis=-1).reshape((-1,20,40))
	sentiment_model_embeddings=sentiment_embedding_model.predict(sX)
	concatenated_embeddings.append(np.concatenate([sales_model_embeddings,rating_model_embeddings,sentiment_model_embeddings],axis=-1))

concatenated_embeddings=np.concatenate(concatenated_embeddings,axis=0)
print concatenated_embeddings.shape
np.savez_compressed('concatenated_embeddings',concatenated_embeddings)