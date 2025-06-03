xhost +local:docker
docker run -it \
           --rm \
           -e DISPLAY=:0 \
           -e QT_X11_NO_MITSHM=1 \
           -v /tmp/.X11-unix:/tmp/.X11-unix \
           -v $HOME/.Xauthority:/root/.Xauthority:rw \
           --gpus all \
           --shm-size=8g \
           -v $(pwd):/workspace \
           bicycle_rl_control