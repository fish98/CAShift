#!/bin/bash

# Terminal 1
minikube delete
minikube start --kubernetes-version=v1.18.12 --driver=docker --container-runtime=containerd
docker ps -a | grep minikube | awk '{print $1}'
tmux new -s record
bash record/record.sh $DOCKER_ID $OUTPUT_DIR $NUMBER

# Terminal 2
source activate $ENV
cd project dir
kubectl apply -k .
k get pods
minikube service wordpress --url

google-chrome $SERVICE_IP

vim normal-performance.py
tmux new -s normal
python daemon.py