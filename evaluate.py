from keras.models import load_model
from auxillary import *

sales_model=load_model('models/sales_model')
final_model=load_model('models/final_model')
print error_final(final_model,sales_model,400,(-1,20,40))