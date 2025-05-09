#!/bin/bash

# Tên tiến trình bạn muốn theo dõi
TARGET="python3 -u -m federated_learning.device_exp.client_app.py 1 10"

while true; do
    # Lấy PID dựa trên câu lệnh chạy
    PID=$(ps -eo pid,cmd | grep "$TARGET" | grep -v grep | awk '{print $1}')

    # Nếu tìm thấy PID
    if [[ ! -z "$PID" ]]; then
        # Lấy %CPU và %MEM
        read CPU MEM <<< $(ps -p $PID -o %cpu,rss --no-headers)

        # Chuyển RSS từ KB -> MB, làm tròn 2 chữ số
        RAM_MB=$(awk "BEGIN {printf \"%.2f\", $MEM/1024}")

        echo "CPU: ${CPU}%, RAM: ${RAM_MB} MB"
    else
        echo "Không tìm thấy tiến trình: $TARGET"
    fi

    sleep 1
done

