#!/bin/bash
set -x

debug=$1
config=$2

run() {
    
    data_source="$1"
    output_prefix="$2"
    mkdir -p $output_prefix

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
	    python3 test.py -s "$data_source" \
		-o $PWD/$output_prefix/$output_prefix_$modelname \
		-m ${modelpath[0]} \
		-n 10  \
		-c $config
	elif [[ "$debug" == "prod" ]]
	then
	    python3 test.py -s "$data_source" \
		-o $PWD/$output_prefix/$output_prefix_$modelname \
		-m ${modelpath[0]} \
		-n 10000  \
		-c theta
	    mail -s "[Theta] $modelname on $output_prefix completed with $status" yadudoc1729@gmail.com < /dev/null
	fi
	status=$?
	echo "Completed -------------------------- $(date)"

	echo -e "\n"
	echo "COMPLETED ##########################################################################################"
    done

    mail -s "[Theta] All inference completed for $output_prefix" yadudoc1729@gmail.com < /dev/null
}


#run '/projects/CVD_Research/datasets/Zinc15_descriptors/*' "Z15_Infer"
run '/projects/CVD_Research/datasets/descriptors-all/savi_descriptors/*' "SAV_Infer"
run '/projects/CVD_Research/datasets/descriptors-all/pubchem128_descriptors/*' "PCH_Infer"
run '/projects/CVD_Research/datasets/descriptors-all/GDB-17_descriptors/*' "G17_Infer"


