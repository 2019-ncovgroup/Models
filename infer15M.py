import pickle
import numpy as np
import pandas as pd
import csv
import argparse

from keras.models import Sequential, Model, load_model
from keras import backend as K
from keras.optimizers import SGD
from sklearn.model_selection import train_test_split
import tensorflow as tf

# hard code args for testing
# args={}
# args['in'] = '/projects/CVD_Research/datasets/15M/xcg.smi.desc.fix'
# args['out'] = 'out_file'
# args['model'] = '/projects/CVD_Research/brettin/March_30/DIR.ml.ADRP-ADPR_pocket1_dock.csv.reg.csv/reg_go.autosave.model.h5

psr = argparse.ArgumentParser(description='inferencing on descriptors')
psr.add_argument('--in',  default='in_file.pkl')
psr.add_argument('--model',  default='model.h5')
psr.add_argument('--out', default='out_file.csv')
args=vars(psr.parse_args())

print(args)

# read the pickle descriptor file
df=pd.read_csv(args['in'])
cols=df.shape[1] - 1
rows=df.shape[0]
samples=np.empty([rows,cols],dtype='float32')
samples=df.iloc[:,1:]
samples=np.nan_to_num(samples)

from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler
scaler = StandardScaler()
df_x = scaler.fit_transform(samples)

# a custom metric was used during training
def r2(y_true, y_pred):
	SS_res =  K.sum(K.square(y_true - y_pred))
	SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
	return (1 - SS_res/(SS_tot + K.epsilon()))

def tf_auc(y_true, y_pred):
	auc = tf.metrics.auc(y_true, y_pred)[1]
	K.get_session().run(tf.local_variables_initializer())
	return auc

def auroc( y_true, y_pred ) :
	score = tf.py_func( lambda y_true, y_pred : roc_auc_score( y_true, y_pred, average='macro', sample_weight=None).astype('float32'),
                        [y_true, y_pred],
                        'float32',
                        stateful=False,
                        name='sklearnAUC' )
	return score


dependencies={'r2' : r2, 'tf_auc' : tf_auc, 'auroc' : auroc }
model = load_model(args['model'], custom_objects=dependencies)
model.summary()
model.compile(loss='mean_squared_error',optimizer=SGD(lr=0.0001, momentum=0.9),metrics=['mae',r2])

predictions=model.predict(df_x)
assert(len(predictions) == rows)

with open (args['out'], "w") as f:
	for n in range(rows):
		print ( "{},{}".format(df.iloc[n,0],predictions[n][0]), file=f)

