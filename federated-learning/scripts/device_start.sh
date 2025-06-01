#!/bin/bash

LOG_DIR=../logs
SERVER_CLIENT_DIR=../federated-learning/device_exp

echo "Current directory: $(pwd)"
if [ ! -d "$LOG_DIR" ]; then
    mkdir -p "$LOG_DIR"
fi
if pgrep -f "python3 -u -m federated_learning.device_exp.server_app" > /dev/null; then
    echo "Killing server_app.py..."
    pkill -f "python3 -u -m federated_learning.device_exp.server_app"
fi

if pgrep -f "python3 -u -m federated_learning.device_exp.client_app" > /dev/null; then
    echo "Killing client_app.py..."
    pkill -f "python3 -u -m federated_learning.device_exp.client_app"
fi


sleep 1


rm $LOG_DIR/server.log $LOG_DIR/client*.log

FLOWER_LOG_LEVEL=DEBUG python3 -u -m federated_learning.device_exp.server_app > $LOG_DIR/server.log 2>&1 &


sleep 3


if [ "$1" = "victim" ]; then
    echo "Starting 10 clients..."
	FLOWER_LOG_LEVEL=DEBUG python3 -u -m federated_learning.device_exp.client_app.py 0 10 > $LOG_DIR/client$i.log 2>&1 &    
for i in {2..9}
    do
        FLOWER_LOG_LEVEL=DEBUG python3 -u -m federated_learning.device_exp.client_app.py $i 10 > $LOG_DIR/client$i.log 2>&1 &
    done
else
    for i in {0..8}
    do
        FLOWER_LOG_LEVEL=DEBUG python3 -u -m federated_learning.device_exp.client_app.py $i 10 > $LOG_DIR/client$i.log 2>&1 &
    done
fi


sleep 1
echo "Server and clients started."
