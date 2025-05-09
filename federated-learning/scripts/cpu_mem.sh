#!/bin/bash
while true; do
    date
    ps -eo pid,pcpu,pmem,cmd | grep 'python3 -u -m federated_learning.device_exp.client_app' | grep -v grep
    echo "-----------------------------------"
    sleep 1
done
