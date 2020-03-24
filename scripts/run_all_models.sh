#!/bin/bash
#set -x

debug=$1
config=$2

models=($(cat models_to_run.txt))
# for i in $(ls /projects/CVD_Research/BoxMirror/drug-screening/ML-models/)
for i in ${models[@]}
do 
    echo "STARTING ##########################################################################################"
    modelname=$(basename $i)
    if [[ $modelname == "3CLpro.reg" ]]
    then
	echo "Skipping $modelname"
	continue
    fi
    if [[ $modelname == "ADRP-P13.bin" ]]
    then
	echo "Skipping $modelname"
	continue
    fi
    
    modelpath=($(ls -rt /projects/CVD_Research/BoxMirror/drug-screening/ML-models/$i/*autosave.model.h5))
    if [[ ${#modelpath[@]} == 0 ]]
    then
	echo "WARNING: $modelname has multiple conflicting models. Skipping"
	echo "${modelpath[*]}"
	continue
    fi

    if [[ ${#modelpath[@]} > 1 ]]
    then
	echo "Multiple : ${modelpath[@]}"
	modelpath=${modelpath[-1]}
	echo "Modelpath set to ${modelpath[0]}"
    fi

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
	python3 test.py -s '/projects/candle_aesp/Descriptors/Enamine_Real/*' \
	    -o $PWD/Enamine_Infer_$modelname \
	    -m ${modelpath[0]} \
            -n 10000  \
	    -c theta
    fi
    status=$?
    echo "Completed -------------------------- $(date)"
    # mail -s "[Theta] $modelname completed with $status" yadudoc1729@gmail.com < /dev/null

    echo -e "\n"
    echo "COMPLETED ##########################################################################################"
done



