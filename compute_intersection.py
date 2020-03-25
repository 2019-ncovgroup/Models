import dask.dataframe as dd
import numpy as np
import pandas as pd
import os

import logging
logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')

import gc
import signal

# hard code args for testing
args={}
args['1']='/lambda_stor/homes/brettin/covid19/ML-models/3CLpro.reg.top1.csv'
args['2']='/lambda_stor/homes/brettin/covid19/ML-models/Enamine_Infer_3CLpro.bin.top1.csv'
args['out'] =  'intersection.top1'

#import argparse
#psr = argparse.ArgumentParser(description='inferencing on descriptors')
#psr.add_argument('--in',  default='in_dir')
#psr.add_argument('--out', default='top1.csv')
#args=vars(psr.parse_args())
print(args)
logging.info("processing {} and {}".format(args['1'], args['2']))


# extra args to set up headers
#kwargs = {'names' : [0, 1, 2] }
kwargs = {}

# parallel read on csv files
logging.info('reading csv files')
df1 = dd.read_csv(args['1'], **kwargs)
df2 = dd.read_csv(args['2'], **kwargs)
logging.info("done reading csv files {} {}".format(args['1'], args['2']))



logging.info('turn it into a pandas dataframe')
df1=df1.compute()
df2=df2.compute()
logging.info("{:,} rows with {:,} elements".format(df1.shape[0], df1.shape[1]))
logging.info("{:,} rows with {:,} elements".format(df2.shape[0], df2.shape[1]))

logging.info("computing intersection on {:,} and {:,} samples".format(df1.shape[0], df2.shape[0]))
s1 = pd.merge(df1, df2, how='inner', on=['2'])
logging.info("done computing intersection resulting in {:,} samples".format(s1.shape[0]))


logging.info ('writing csv')
s1.to_csv(args['out'])
logging.info('done writing csv')

# for some reason, it takes forever for the program
# to exit, am assuming it garbage collection, but can't prove it yet.
# gc.set_threshold(0)
# os.kill(os.getpid(), signal.SIGTERM)


