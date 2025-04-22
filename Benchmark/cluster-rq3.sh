#/bin/bash

source activate logae

algorithm_list=("gini" "zol" "kmst")
percentage_list=("0.1" "0.3" "0.5")

for algorithm in "${algorithm_list[@]}"; do
    echo Algorhthm $algorithm

    for percentage in "${percentage_list[@]}"; do
        echo Percentage $percentage

        # step 1: change yaml
        python change_yaml_rq3.py $algorithm $percentage

        # step 2: run retrain
        python main.py --config=configs/retrain-rq3-cluster.yaml
    done
done

python change_yaml_rq3.py gini 1
python main.py --config=configs/retrain-rq3-cluster.yaml