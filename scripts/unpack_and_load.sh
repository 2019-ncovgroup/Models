#!/bin/bash

mode=$1
env_name=$2
tar_file=$3
dest_path=$4
env_root=$dest_path/$env_name

log=$JOBNAME
nid=$(uname -n)


test_from_tmp() {
    echo "Running temp-------------------"
    pushd .

    # Clear the dest paths
    cd $dest_path

    if [ -d $env_root ]; then
	echo "Untar path $dest_path/$env_name exists. Wiping..."
	rm -rf $env_root
    fi

    # Remake the dest path with dirs for unpacking
    echo "Creating $dest_path/$env_name"
    mkdir $env_root

    pvar=$( { time cp $tar_file $dest_path/ ; } 2>&1 )
    var=$(echo $pvar)
    echo "Time_to_copy_to_tmp, $nid, $env_root, $var"

    nodename=$(uname -n)
    tmp_tar_file=$dest_path/$(basename $tar_file)
    echo "Path : $tmp_tar_file"
    pvar=$( { time tar -xzf $tmp_tar_file -C $env_root ; } 2>&1 )
    var=$(echo $pvar)
    echo "Time_to_untar, $nid, $env_root, $var"
    
    popd
}

test_from_shared() {
    echo "Running shared mode--------------"
    pushd .

    source ~/anaconda3/bin/activate
    conda activate $env_name
    echo "Activated env : $env_name"
}


if [ "$mode" == "tmp" ]; then
    test_from_tmp

elif [ "$mode" == "shared" ]; then
    test_from_shared

else
    echo "Unknown mode $mode requested"
fi
