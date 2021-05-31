#!/bin/bash

tree_dirs=./trees/tree*/hard_queries/
index=0
for dir in $tree_dirs 
do 
    echo ${dir}
    for query_file in $dir/*
    do
        dest="./queries/query${index}"
        cp $query_file $dest
	index=$((index + 1))
    done
done
