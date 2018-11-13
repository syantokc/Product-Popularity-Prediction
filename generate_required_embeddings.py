from keras.models import load_model, Model
import numpy as np
from auxillary_final import *

sales_trainX,rating_trainX,sentiment_trainX,Y,diff,ids=getAllEvaluationData(400)

sales_model=load_model('models/sales_model_1')
# sales_model.summary()
sales_embedding_model=Model(inputs=sales_model.input,outputs=sales_model.get_layer('lstm_1').output)
rating_model=load_model('models/rating_model_1')
# rating_model.summary()
rating_embedding_model=Model(inputs=rating_model.input,outputs=rating_model.get_layer('lstm_2').output)
sentiment_model=load_model('models/sentiment_model_1')
# sentiment_model.summary()
sentiment_embedding_model=Model(inputs=sentiment_model.input,outputs=sentiment_model.get_layer('lstm_3').output)

sales_embeddings=sales_embedding_model.predict(sales_trainX.reshape((-1,20,40)))
rating_embeddings=rating_embedding_model.predict(rating_trainX.reshape((-1,20,40)))
sentiment_embeddings=sentiment_embedding_model.predict(sentiment_trainX.reshape((-1,20,40)))

print sales_embeddings.shape
print rating_embeddings.shape
print sentiment_embeddings.shape

np.savez_compressed('required/sales_embeddings',sales_embeddings)
np.savez_compressed('required/rating_embeddings',rating_embeddings)
np.savez_compressed('required/sentiment_embeddings',sentiment_embeddings)
f=open('required/ids.txt','w')
for i in ids:
	f.write(i+'\n')
f.close()