from auxillary_final import *
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, GRU, Merge
from keras.layers.wrappers import Bidirectional, TimeDistributed
from keras.optimizers import SGD
import numpy as np

# full_output(None,400,'')

U=GRU

sales_model=Sequential()
sales_model.add(U(100,input_shape=(20,40)))
#sales_model.add(U(100))

rating_model=Sequential()
rating_model.add(U(100,input_shape=(20,40)))
#rating_model.add(U(100))

titlepos_model=Sequential()
titlepos_model.add(U(100,input_shape=(20,40)))
#titlepos_model.add(U(100))

titleneg_model=Sequential()
titleneg_model.add(U(100,input_shape=(20,40)))
#titleneg_model.add(U(100))

textpos_model=Sequential()
textpos_model.add(U(100,input_shape=(20,40)))
#textpos_model.add(U(100))

textneg_model=Sequential()
textneg_model.add(U(100,input_shape=(20,40)))
#textneg_model.add(U(100))

model=Sequential()
model.add(Merge([sales_model,rating_model,titlepos_model,titleneg_model,textpos_model,textneg_model],mode='concat'))
model.add(Dense(150,activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(50,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1))



model_name='GRU'
model.load_weights('models/combined_'+model_name)
error=full_output(model,400,model_name)
print 'Error for',model_name,':',error
