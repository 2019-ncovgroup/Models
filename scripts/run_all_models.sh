#!/bin/bash
#set -x

debug=$1
config=$2

models=($(cat models_to_run.txt))
# for i in $(ls /projects/CVD_Research/BoxMirror/drug-screening/ML-models/)
for i in ${models[@]}
do 
    echo "STARTING ##########################################################################################"
    modelname=$(basename $(dirname $i))    
    modelpath=$i

    echo "Starting processing of $modelname at $(date)"
    echo "Running $modelname with path : $modelpath"

    if [[ "$debug" == "skip" ]]
    then
	echo "Debug skip"

    elif [[ "$debug" == "debug" ]]
    then
	python3 test.py -s '/projects/candle_aesp/Descriptors/Enamine_Real/*' \
	    -o $PWD/Enamine_Infer_$modelname \
	    -m ${modelpath[0]} \
            -n 10  \
	    -c $config
    elif [[ "$debug" == "prod" ]]
    then
	python3 test.py -s '/projects/CVD_Research/datasets/Enamine_Real/*' \
	    -o $PWD/Enamine_Infer_$modelname \
	    -m ${modelpath[0]} \
            -n 10000  \
	    -c theta
	mail -s "[Theta] $modelname completed with $status" yadudoc1729@gmail.com < /dev/null
    fi
    status=$?
    echo "Completed -------------------------- $(date)"

    echo -e "\n"
    echo "COMPLETED ##########################################################################################"
done



