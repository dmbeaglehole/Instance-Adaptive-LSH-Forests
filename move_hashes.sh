#!/bin/bash

tree_dirs=./trees/tree*/
for dir in $tree_dirs 
do 
    index=$(echo $dir | tr -dc '0-9')
    hash_file="${dir}hashes"
    dest="./hashes/hashes${index}"
    cp $hash_file $dest
done
