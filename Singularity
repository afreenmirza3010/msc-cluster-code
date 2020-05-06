Bootstrap: docker
From: tensorflow/tensorflow:latest-gpu-py3
Stage: build

%post
    apt update -y
    apt upgrade -y
    pip install ipython
    pip install https://github.com/huynhngoc/deoxys/archive/Multiple-losses.zip
    pip install comet-ml
    pip install scikit-image

%environment
    export KERAS_MODE=TENSORFLOW
    
