#!/bin/bash

#  run_sript.sh
#  Created by Doris V on 9/9/2026.

seed_arr=($(seq 1 1 10))
noise_group_arr=(0.1 1)
noise_common_arr=(0.1 1)
noise_independent_arr=(0.1 1)

for seed in "${seed_arr[@]}"
do
    for noise_group in "${noise_group_arr[@]}"
    do
        for noise_common in "${noise_common_arr[@]}"
        do
            for noise_independent in "${noise_independent_arr[@]}"
            do
            
                python main.py --seed=$seed --noise_group=$noise_group --noise_common=$noise_common --noise_independent=$noise_independent
                
            done
        done
    done
done
