import pickle
import csv
import glob
import argparse
import os
import numpy as np

def convert(csvfile, pklfile):

    print("In convert")
    line_count = 0
    bad_count = 0
    bad_line = None
    data = {}
    smiles = []
    with open(csvfile) as f:
        for line in f:
            line_count += 1            
            dset, ident, smile, *desc = line.strip().split(',')
            smiles.extend([smile])
            desc_clean = ['nan' if i == '' else i for i in desc]            
            try:                

                descriptor = np.array(desc_clean, dtype=np.float32)
                if smile in data:
                    older = data[smile][1]
                    if (older == descriptor).all():
                        print("Match")
                    else:
                        print("Nonmatching")

                data[smile] = ([ident], descriptor)
                print(len(data))
            except Exception as e:
                bad_count += 1
                bad_line = desc_clean
                print("Caught exception : {}".format(e))
    
    
    print("Smiles : {}, uniq: {}".format(len(smiles), len(set(smiles))))
    print("Data : {}".format(len(data)))
    print("Caught {}/{} bad lines/total lines".format(bad_count, line_count))
    with open(pklfile, 'wb') as f:
        pickle.dump(data, f)

    

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--version",
                        help="Print Endpoint version information")
    parser.add_argument("-d", "--debug", action='store_true',
                        help="Enables debug logging")
    parser.add_argument("-n", "--num_files", default=10000,
                        help="Number of files to load and run from each descriptor dir. Default=all, if set to 0 the entire file will be used")
    parser.add_argument("-s", "--smile_dir", required=True,
                        help="File path to the descriptor directories. Name retained to not break scripts")
    parser.add_argument("-o", "--outdir", default="outputs",
                        help="Output directory. Default : outputs")
    args = parser.parse_args()


    print("Smiles glob :", args.smile_dir)
    for descriptor_csv in glob.glob(args.smile_dir):
        b = os.path.basename(descriptor_csv)
        output = "{}/{}.pkl".format(args.outdir, b.rstrip('.csv'))
        print(descriptor_csv, output)
        convert(descriptor_csv, output)

    
