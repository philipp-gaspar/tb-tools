echo ' '
echo 'Start Docker Container'

docker run --gpus all \
    --network host \
    -v ${HOME}/BRICS-TB/data-schenzen/raw:/home/philipp.gaspar/BRICS-TB/data-schenzen/data \
    -v ${HOME}/BRICS-TB/tb-tools/experiments:/home/philipp.gaspar/BRICS-TB/tb-tools/experiments \
    -v ${HOME}/BRICS-TB/tb-tools:/home/philipp.gaspar/BRICS-TB/tb-tools \
    -it -u $(id -u):$(id -g) philippgaspar/tb-tools:latest-gpu \
    python ${HOME}/BRICS-TB/tb-tools/src/cnn_model.py

echo ' '
echo 'Finish Docker Container'