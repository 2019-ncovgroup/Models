import pickle
import numpy as np
import pandas as pd
import csv
import sys

import logging
logging.basicConfig(
	format='%(asctime)s %(levelname)-8s %(message)s',
	level=logging.INFO,
	datefmt='%Y-%m-%d %H:%M:%S')

infile=sys.argv[1]
logging.info("processing {}".format(infile))


logging.info("reading pkl into pandas")
pf=open(infile, 'rb')
data=pickle.load(pf)
df=pd.DataFrame(data).transpose()
df.dropna(how='any', inplace=True)
pf.close()



logging.info("building np array from pkl file")
cols=len(df.iloc[0][1])
rows=df.shape[0]
samples=np.empty([rows,cols],dtype='float32')

for i in range(rows):
	a=df.iloc[i,1]
	samples[i]=a

nans=0
total=0

logging.info("counting nans")
for i in range(samples.shape[0]):
	for j in range(samples.shape[1]):
		if np.isnan(samples[i,j]):
			nans=nans+1
		total=total+1

logging.info(("nans {:,}, total {:,}".format(nans,total)))
