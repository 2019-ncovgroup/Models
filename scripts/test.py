# from candle_apps.candle_apps import run_intranode
# from candle_apps.candle_apps import ModelInferer
#from candle_apps.candle_node_local import run_local
#import pandas as pd
import os
import glob
import argparse
import traceback
from covid_screen.reg_go_infer import reg_go_infer, reg_go_infer_csv
import parsl
import pickle



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--version",
                        help="Print Endpoint version information")
    parser.add_argument("-d", "--debug", action='store_true',
                        help="Enables debug logging")
    parser.add_argument("-n", "--num_files", default=10000,
                        help="Number of files to load and run. Default=all, if set to 0 the entire file will be used")
    parser.add_argument("-s", "--smile_dir", default=".",
                        help="File path to the smiles csv file")
    parser.add_argument("-b", "--batch_size", default="4",
                        help="Size of the batch of smiles to send to each node for processing. Default=4, should be 10K")
    parser.add_argument("-o", "--outdir", default="outputs",
                        help="Output directory. Default : outputs")
    parser.add_argument("-m", "--model", required=True,
                        help="Specify full path to model to run")
    parser.add_argument("--selector", default="pkl",
                        help="Specify file extension to use for input files")
    parser.add_argument("-c", "--config", default="local",
                        help="Parsl config defining the target compute resource to use. Default: local")
    args = parser.parse_args()

    #for smile_dir in glob.glob(args.smile_dir):
    #    print(smile_dir)
    #exit(0)
    print(f"Loading pkl files from {args.smile_dir}")

    if args.config == "local":
        from parsl.configs.htex_local import config
        from parsl.configs.htex_local import config
        config.executors[0].label = "Foo"
        config.executors[0].max_workers = 1
    elif args.config == "theta":
        from theta import config
        print("Loading theta config")
    elif args.config == "theta_test":
        from theta_test import config
    elif args.config == "comet":
        from comet import config

    # Most of the app that hit the timeout will complete if retried.
    # but for this demo, I'm not setting retries.
    # config.retries = 2
    parsl.load(config)

    parsl_runner = {}
    parsl_runner['pkl'] = parsl.python_app(reg_go_infer)
    parsl_runner['csv'] = parsl.python_app(reg_go_infer_csv)

    if args.debug:
        parsl.set_stream_logger()


    os.makedirs(args.outdir, exist_ok=True)

    all_smile_dirs = glob.glob(args.smile_dir)
    counter = 0

    batch_futures = {}
    for smile_dir in all_smile_dirs:
        print("Processing smile_dir: {} {}/{}".format(smile_dir, counter, len(all_smile_dirs)))
        counter+=1
        batch_futures[smile_dir] = []

        outdir = "{}/{}".format(args.outdir, os.path.basename(smile_dir))
        os.makedirs(outdir, exist_ok=True)
        os.makedirs(outdir + '/logs' , exist_ok=True)

        for input_file in os.listdir(smile_dir)[:int(args.num_files)]:
            fname = os.path.basename(input_file)
            if input_file.endswith('.pkl') and args.selector == "pkl":
                out_file = "{}/{}".format(outdir, 
                                          fname.replace('.pkl', '.csv'))
                log_file = "{}/logs/{}".format(outdir, 
                                               fname.replace('.pkl', '.log'))
                kind = 'pkl'
                
            elif input_file.endswith('csv') and args.selector == "csv":
                out_file = "{}/{}".format(outdir, 
                                          fname.replace('.csv', '.out.csv'))
                log_file = "{}/logs/{}".format(outdir, 
                                               fname.replace('.csv', '.log'))                
                kind = 'csv'
            else:                
                continue
        
            if os.path.exists(out_file):
                # Skip compute entirely if output file already exists
                continue

            input_file_path = f"{smile_dir}/{input_file}"
            x = parsl_runner[kind](input_file_path,
                                   args.model, #'/projects/candle_aesp/yadu/Models/scripts/agg_attn.autosave.model.h5', #3CLpro.reg                            
                                   # '/projects/candle_aesp/yadu/Models/ADRP-P1.reg/descriptor_headers',
                                   '/projects/candle_aesp/yadu/Models/ADRP-P1.reg/descriptor_headers.csv',
                                   '/projects/candle_aesp/yadu/Models/ADRP-P1.reg/training_headers.csv',
                                   out_file,
                                   log_file)

            batch_futures[smile_dir].append(x)

        # Waiting for all futures
        print("Waiting for all futures from {}".format(smile_dir))

        for i in batch_futures[smile_dir]:
            try:
                x = i.result()

            except Exception as e:
                print("Exception : {} Traceback : {}".format(e, traceback.format_exc()))
                print(f"Chunk {i} failed")
        print(f"Completed {smile_dir}")    

    print("All done!")

