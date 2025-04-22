#!/bin/bash

# CVE_ID="CWE-200"
CVE_ID=$1
RECORD_TIME=15
FILE_NUMBER=1
CONTAINER_ID=$(docker ps | grep minikube | awk '{print $1}')

OUTPUT_DIR="/CAShift/Dataset/output/attack/$CVE_ID"

if [ ! -d "$OUTPUT_DIR" ]; then
    echo "Creating Output Dir"
    mkdir -p $OUTPUT_DIR
else
    echo "Output Dir Already Exists"
fi

# get pod id
POD_ID=$(kubectl get pods | grep wordpress | awk '{print $1}')

tmux new-session -d -s attack
tmux new-session -d -s record

# setup env
docker cp ./k8spider/bin/k8spider $CONTAINER_ID:/home/docker/
KUBERNETES_SERVICE_HOST=$(minikube service wordpress --url | grep -oP 'http://\K[\d.]+')

for i in {1..100}; do   

    # collect system log
    echo "Collecting $i ..."
    record_cmd="sudo sysdig -v -b -p \"%evt.rawtime %user.uid %proc.pid %proc.name %thread.tid %syscall.type %evt.dir %evt.args\" -w $OUTPUT_DIR/$FILE_NUMBER.scap container.id=$CONTAINER_ID"
    tmux send -t "record" "$record_cmd" ENTER

    # sleep to ensure the record is started and log start
    sleep 2
    kubectl exec -it $POD_ID -c wordpress -- logger 'Attack Start'
    
    terminate_cmd="kubectl exec -it $POD_ID -c wordpress -- logger 'Attack Stop' && sleep 0.5 && tmux send -t 'record' 'C-c' ENTER"
    
    attack_cmd="minikube ssh 'export KUBERNETES_SERVICE_HOST=$KUBERNETES_SERVICE_HOST && /home/docker/k8spider all && /home/docker/k8spider wild' || true && $terminate_cmd"
    # k8spider neighbor could be used
    tmux send -t "attack" "$attack_cmd" ENTER

    FILE_NUMBER=$((FILE_NUMBER + 1))
    sleep 10 && echo "Finish $FILE_NUMBER"
done

# clean up
tmux kill-session -t attack
tmux kill-session -t record

# minikube delete