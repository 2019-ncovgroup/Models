High-Throughput Inference Pipeline
==================================

If you are modding the configs to run this and run into issues mail Yadu at <yadudoc1729@gmail.com>.

This pipeline is designed to take directories containing pickle files and apply an inference task on each.
The pickle files follow this structure::

   { <KEY: SMILE_STRING> : LIST[ <STRING_DRUG_ID> : <NUMPY_FLAT_ARRAY> ] .... } 

Each inference task is a python function that takes paths to the model file (.h5), pickle files and
produce a corresponding log file and csv file that contains the output.


Setting up your Theta env.
--------------------------

Step 1: Setup conda env

>>> module load miniconda-3/latest
>>> conda create -p /projects//candle_aesp/yadu/candle_inferpy3.7 --clone $CONDA_PREFIX
>>> conda activate /projects/candle_aesp/yadu/candle_inferpy3.7

Step 2: Install all the required packaged. Most packages already come baked in with the base conda

>>> git clone https://github.com/Parsl/parsl.git
>>> cd parsl
>>> pip install .

>>> pip install conda-pack

Step 3: [Optional] Only for performance at scale.

>>> conda-pack -n candle_inferpy3.7 -o /tmp/candle_inferpy3.7.tar.gz


Running the pipeline
--------------------

Here are supported options :

.. code-block:: python

    python3 test.py -h

    usage: test.py [-h] [-v VERSION] [-d] [-n NUM_FILES] [-s SMILE_DIR]
		   [-b BATCH_SIZE] [-o OUTDIR] -m MODEL [-c CONFIG]

    optional arguments:
      -h, --help            show this help message and exit
      -v VERSION, --version VERSION
			    Print Endpoint version information
      -d, --debug           Enables debug logging
      -n NUM_FILES, --num_files NUM_FILES
			    Number of files to load and run. Default=all, if set
			    to 0 the entire file will be used
      -s SMILE_DIR, --smile_dir SMILE_DIR
			    File path to the smiles csv file
      -b BATCH_SIZE, --batch_size BATCH_SIZE
			    Size of the batch of smiles to send to each node for
			    processing. Default=4, should be 10K
      -o OUTDIR, --outdir OUTDIR
			    Output directory. Default : outputs
      -m MODEL, --model MODEL
			    Specify full path to model to run
      -c CONFIG, --config CONFIG
			    Parsl config defining the target compute resource to
			    use. Default: local



For example you can invoke the pipeline like this:

>>> python3 test.py -s /projects/candle_aesp/Descriptors/Enamine_Real/2019q3-4_Enamine_REAL_10_descriptors -o $PWD/Enamine_10 -n 10000 -c theta -m <Path to model file>


Production runs are doing using the `run_all_models.sh` script. It launches runs for each ML model available here `/projects/CVD_Research/BoxMirror/drug-screening/ML-models/`.


Debugging
---------

For debugging, you want to run with a limited number of files, say 2 and run locally on the login node to avoid queueing times.
Here's an example 

>>> python3 test.py -s /projects/candle_aesp/Descriptors/Enamine_Real/2019q3-4_Enamine_REAL_10_descriptors -o $PWD/Enamine_Testing -n 2 -c local -m <Path to model file>










