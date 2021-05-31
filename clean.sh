#!/bin/bash
num_trees=300
for i in $(seq 0 $num_trees); do 
    echo $i
    rm trees/tree$i/dists/*
    rm trees/tree$i/hard_queries/*
done

rm dataset.txt
rm pi_init.txt
rm trees/tree*/hashes
rm trees/tree*/collision_probs/*
rm nohup.out
rm slurm-*.out
rm slurm_files/tree*/*
rm core*
