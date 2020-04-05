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
#args={}
#args['in']='3CLpro.reg'
#args['in']='Enamine_Infer_3CLpro.bin'
#args['out'] = args['in'] + '.top1'

import argparse
psr = argparse.ArgumentParser(description='inferencing on descriptors')
psr.add_argument('--in',  default='in_dir')
psr.add_argument('--out', default='top1.csv')
psr.add_argument('--perc', default=1, type=int)

args=vars(psr.parse_args())
print(args)

logging.info('processing '+ args['in'])


# get the list of files recursively from root dir
l=[]
for root, subdirs, files in os.walk(args['in']):
        for f in files:
                if f.endswith('.csv'):
                        l.append(root + '/' + f)

logging.info ("found {} files".format(str(len(l))))


# extra args to set up headers
kwargs = {'names' : [0, 1, 2] }

# make l small for testing
# l=l[0:20000]

# parallel read on csv files
logging.info('reading csv files')
df = dd.read_csv(l, **kwargs)
logging.info("done reading {:,} csv files".format(len(l)))


logging.info('turn it into a pandas dataframe')
df=df.compute()
logging.info("{:,} rows with {:,} elements".format(df.shape[0], df.shape[1]))


p=1-(args['perc']/100)
r=str(100-args['perc']) + '%'
logging.info("computing percentiles on pandas dataframe using {}".format(p))
d=df.describe(percentiles=[p])
logging.info("computed percentiles on pandas dataframe")
logging.info("cutoff score for top {}% {:,}".format(args['perc'], d.at[r,1]))


logging.info("identifying top {}%".format(args['perc']))
r=str(100-args['perc']) + '%'
t = df[df[1] > d.at[r,1]]
logging.info("identified {:,} compounds in the top {}%".format(t.shape[0], args['perc']))


logging.info ('writing csv')
t.to_csv(args['out'])
logging.info('done writing csv')

# for some reason, it takes forever for the program
# to exit, am assuming it garbage collection, but can't prove it yet.
gc.set_threshold(0)
os.kill(os.getpid(), signal.SIGTERM)


