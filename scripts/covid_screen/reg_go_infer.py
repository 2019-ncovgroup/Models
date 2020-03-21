import logging

def set_file_logger(filename: str, name: str = 'candle', level: int = logging.DEBUG, format_string = None):
    """Add a stream log handler.

    Args:
        - filename (string): Name of the file to write logs to
        - name (string): Logger name
        - level (logging.LEVEL): Set the logging level.
        - format_string (string): Set the format string

    Returns:
       -  None
    """
    if format_string is None:
        format_string = "%(asctime)s.%(msecs)03d %(name)s:%(lineno)d [%(levelname)s]  %(message)s"

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    handler = logging.FileHandler(filename)
    handler.setLevel(level)
    formatter = logging.Formatter(format_string, datefmt='%Y-%m-%d %H:%M:%S')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger

def reg_go_infer(pkl_file, model, descriptor_headers, training_headers, out_file, log_file_path):
    import time
    start = time.time()
    import pickle
    import numpy as np
    import pandas as pd
    import csv
    import argparse
    import os

    from keras.models import Sequential, Model, load_model
    from keras import backend as K
    from keras.optimizers import SGD
    from sklearn.model_selection import train_test_split
    loadtime = time.time() - start

    logger = set_file_logger(log_file_path)
    logger.info("Start================================================== on {}".format(os.uname()))
    """
    psr = argparse.ArgumentParser(description='inferencing on descriptors')
    psr.add_argument('--in',  default='in_file.pkl')
    psr.add_argument('--model',  default='model.h5')
    psr.add_argument('--dh',  default='descriptor_headers.csv')
    psr.add_argument('--th',  default='training_headers.csv')
    psr.add_argument('--out', default='out_file.csv')
    args=vars(psr.parse_args())

    print(args)
    """

    # get descriptor and training headers
    # the model was trained on 1613 features
    # the new descriptor files have 1826 features
    logger.info(f"Pkl file : {pkl_file}")
    logger.info(f"model : {model}")
    logger.info(f"descriptor_headers : {descriptor_headers}")
    logger.info(f"training_headers : {training_headers}")
    logger.info(f"out_file : {out_file}")
    logger.info(f"Python libs loading time : {loadtime}")

    with open (descriptor_headers) as f:
        reader = csv.reader(f, delimiter=",")
        drow = next(reader)
        drow = [x.strip() for x in drow]

    tdict={}
    for i in range(len(drow)):
        tdict[drow[i]]=i
 

    f.close()
    del reader

    with open (training_headers) as f:
        reader = csv.reader(f, delimiter=",")
        trow = next(reader)
        trow = [x.strip() for x in trow]

    f.close()
    del reader


    # read the pickle descriptor file

    pf=open(pkl_file, 'rb')
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

    samples=np.nan_to_num(samples)

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

    # a custom metric was used during training

    def r2(y_true, y_pred):
        SS_res =  K.sum(K.square(y_true - y_pred))
        SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
        return (1 - SS_res/(SS_tot + K.epsilon()))

    logger.info("Loading model")
    dependencies={'r2' : r2 }
    model = load_model(model, custom_objects=dependencies)
    model.summary()
    model.compile(loss='mean_squared_error',optimizer=SGD(lr=0.0001, momentum=0.9),metrics=['mae',r2])

    logger.info("Predicting")
    predictions=model.predict(df_x)
    assert(len(predictions) == rows)

    logger.info("Starting write out")
    tmp_file = "/tmp/{}".format(os.path.basename(out_file))
    with open (tmp_file, "w") as f:
        for n in range(rows):
            print ("{},{},{}".format(df.iloc[n,0][0],predictions[n][0],df.index[n] ), file=f)
    os.system(f'cp {tmp_file} {out_file}')

    logger.info("All done")
    logger.handlers.pop()

    delta = time.time() - start
    logger.info("Total time : {:8.3f}s".format(delta))
    return out_file

if __name__ == '__main__':
    
    reg_go_infer('/projects/candle_aesp/Descriptors/Enamine_Real/2019q3-4_Enamine_REAL_01_descriptors/2019q3-4_Enamine_REAL_01.smi.chunk-0-10000.pkl',
                 '/projects/candle_aesp/yadu/Models/ADRP-P1.reg/agg_attn.autosave.model.h5',
                 '/projects/candle_aesp/yadu/Models/ADRP-P1.reg/descriptor_headers',
                 '/projects/candle_aesp/yadu/Models/ADRP-P1.reg/training_headers',
                 '/projects/candle_aesp/yadu/Models/scripts/out1.csv',
                 '/projects/candle_aesp/yadu/Models/scripts/out1.log'
                 )
