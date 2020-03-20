import pickle
import numpy as np
import pandas as pd
import csv

from keras.models import Sequential, Model, load_model
from keras import backend as K
from keras.optimizers import SGD
from sklearn.model_selection import train_test_split


# hard code args for testing
args={}
args['evaluate'] = False
args['in'] = '2019q3-4_Enamine_REAL_01.smi.chunk-0-10000.pkl'
args['dh'] = 'descriptor_headers'
args['th'] = 'training_headers'
args['ev'] = 'adrp-p1.csv'


# get descriptor and training headers
with open (args['dh']) as f:
	reader = csv.reader(f, delimiter=",")
	drow = next(reader)
	drow = [x.strip() for x in drow]

tdict={}
for i in range(len(drow)):
	tdict[drow[i]]=i

f.close()
del reader

with open (args['th']) as f:
	reader = csv.reader(f, delimiter=",")
	trow = next(reader)
	trow = [x.strip() for x in trow]


f.close()
del reader


pf=open(args['in'], 'rb')
data=pickle.load(pf)
df=pd.DataFrame(data).transpose()
df.dropna(how='any', inplace=True)
pf.close()

# build np array from pkl file
cols=len(df.iloc[0][1])
rows=df.shape[0]
samples=np.empty([rows,cols],dtype='float32')

for i in range(rows):
        a=df.iloc[i,1]
        samples[i]=a

# build np array with reduced feature set
reduced=np.empty([rows,len(trow)],dtype='float32')
i=0
for h in trow:
	reduced[:,i]=samples[:,tdict[h] ]
	i=i+1

# del samples

from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler
scaler = StandardScaler()
df_x = scaler.fit_transform(reduced)



df_toss = (pd.read_csv(args['ev'],nrows=1).values)
PL = df_toss.size

def load_data():
	data_path = args['ev']
	df = (pd.read_csv(data_path,skiprows=1).values).astype('float32')
	df_y = df[:,0].astype('float32')
	df_x = df[:, 1:PL].astype(np.float32)
	scaler = StandardScaler()
	df_x = scaler.fit_transform(df_x)
	X_train, X_test, Y_train, Y_test = train_test_split(df_x, df_y, test_size= 0.20, random_state=42)
	print('x_train shape:', X_train.shape)
	print('x_test shape:', X_test.shape)
	return X_train, Y_train, X_test, Y_test


def r2(y_true, y_pred):
	SS_res =  K.sum(K.square(y_true - y_pred))
	SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
	return (1 - SS_res/(SS_tot + K.epsilon()))


dependencies={'r2' : r2 }
model = load_model('agg_attn.autosave.model.h5', custom_objects=dependencies)
model.summary()
model.compile(loss='mean_squared_error',optimizer=SGD(lr=0.0001, momentum=0.9),metrics=['mae',r2])

if args['evaluate']:
	X_train, Y_train, X_test, Y_test = load_data()
	model.evaluate(X_test, Y_test)

predictions=model.predict(df_x)
assert(len(predictions) == rows)

# for n in range(rows):
# 	print ( "{},{},{}".format( ) )

