from parsl.config import Config
from parsl.executors import HighThroughputExecutor

from parsl.launchers import JsrunLauncher
from parsl.providers import LSFProvider

from parsl.addresses import address_by_interface

NODES=40
site_specifics = {
    'training_headers' : '/gpfs/alpine/proj-shared/med110/yadu/Models/ADRP-P1.reg/training_headers.csv',
    'descriptor_headers': '/gpfs/alpine/proj-shared/med110/yadu/Models/ADRP-P1.reg/descriptor_headers.csv',
}

config = Config(
    executors=[
        HighThroughputExecutor(
            label='Summit_HTEX',
            # On Summit ensure that the working dir is writeable from the compute nodes,
            # for eg. paths below /gpfs/alpine/world-shared/
            max_workers=1,
            #working_dir='YOUR_WORKING_DIR_ON_SHARED_FS',
            # address=address_by_interface('ib0'),  # This assumes Parsl is running on login node
            prefetch_capacity=2,
            heartbeat_period=60,
            heartbeat_threshold=120,
            worker_port_range=(50000, 55000),
            provider=LSFProvider(
                # Jsrun manual -> https://www.ibm.com/support/knowledgecenter/en/SSWRJV_10.1.0/jsm/10.3/base/jsrun.html
                launcher=JsrunLauncher(overrides='--exit_on_error 0 -r 6 -n {} --tasks_per_rs=1 -c 7 --gpu_per_rs=1 --bind=rs '.format(NODES*6)),
                walltime="02:00:00",
                nodes_per_block=NODES,
                init_blocks=1,
                max_blocks=1,
                worker_init='source ~/candle_setup.sh; export PYTHONIOENCODING="UTF-8"', 
                # Input your worker environment initialization commands
                # scheduler_options='#BSUB -P MED110',
                #scheduler_options='#BSUB -q killable \nmodule load job-step-viewer',
                scheduler_options='module load job-step-viewer',
                project='MED110',
                cmd_timeout=60
            ),
        )

    ],
)
