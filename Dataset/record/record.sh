#!/bin/bash

CONTAINER_ID=$1
OUTPUT_DIR=$2
LOG_NUMBER=$3
RECORD_TIME=15 # record 15 seconds
FILE_NUMBER=1

if [ "$#" -ne 3 ]; then
    echo "Usage: $0 CONTAINER_ID OUTPUT_DIR LOG_NUMBER"
    echo "Example: $0 normal_container /output/ 1500"
    exit 1
fi

if [ ! -d "$OUTPUT_DIR" ]; then
    echo "Creating Output Dir"
    mkdir -p $OUTPUT_DIR
else
    echo "Output Dir Already Exists"
fi

while true
do
    sudo sysdig -v -b -p "%evt.rawtime %user.uid %proc.pid %proc.name %thread.tid %syscall.type %evt.dir %evt.args" -M $RECORD_TIME -w $OUTPUT_DIR/$FILE_NUMBER.scap container.id=$CONTAINER_ID
    # echo $FILE_NUMBER
    FILE_NUMBER=$((FILE_NUMBER + 1))
    sleep 0.5
    # if $FILE_NUMBER = 1500 then break
    if [ $FILE_NUMBER -eq $LOG_NUMBER ]; then
        break
    fi
done