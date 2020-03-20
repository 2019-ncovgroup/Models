import pickle
import numpy as np
import pandas as pd

from keras.models import Sequential, Model, load_model
from keras import backend as K
from keras.optimizers import SGD

args={}
args['evaluate'] = False
args['in'] = '2019q3-4_Enamine_REAL_01.smi.chunk-0-10000.pkl'


pf=open(args['in'], 'rb')
data=pickle.load(pf)
df=pd.DataFrame(data).transpose()
df.dropna(how='any', inplace=True)
pf.close()

cols=len(df.iloc[0][1])
rows=df.shape[0]
samples=np.empty([rows,cols],dtype='float32')

for i in range(rows):
	a=df.iloc[i,1]
	samples[i]=a


def r2(y_true, y_pred):
	SS_res =  K.sum(K.square(y_true - y_pred))
	SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
	return (1 - SS_res/(SS_tot + K.epsilon()))


dependencies={'r2' : r2 }
model = load_model('agg_attn.autosave.model.h5', custom_objects=dependencies)
model.summary()
model.compile(loss='mean_squared_error',optimizer=SGD(lr=0.0001, momentum=0.9),metrics=['mae',r2])

if args['evaluate']:
	model.evaluate(X_test, Y_test)

predictions=model.predict(samples)

