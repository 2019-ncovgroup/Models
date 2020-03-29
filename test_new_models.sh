#!/bin/bash

in_dir=$1
test_data="2019q3-4_Enamine_REAL_12.smi.chunk-9970000-9980000.pkl"
for n in $(find $in_dir/ -name "*autosave.model.h5" ) ;do 
	echo $n
	d=$(dirname $n) 
	o=$(basename $d ) 
	python ./ADRP-P1.reg/reg_go_infer.py --in $test_data --dh ADRP-P1.reg/descriptor_headers.csv --th ADRP-P1.reg/training_headers.csv --model $n --out $o.pred

done

wc *.pred

