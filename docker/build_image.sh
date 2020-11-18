#!bin/bash

docker build --build-arg USERNAME=$(whoami) \
             --build-arg UID=$(id -u) \
             --build-arg GID=$(id -g) \ 
             -t philippgaspar/tb-brics:latest-gpu .