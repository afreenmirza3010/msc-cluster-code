Bootstrap: docker
From: tensorflow/tensorflow:latest-gpu-py3
Stage: build

%post
    apt update -y
    apt upgrade -y
    pip install ipython
    pip install http://github.com/huynhngoc/deoxys/archive/master.zip
    pip install tensorflow==2.0.0
    pip install comet-ml
    pip install scikit-image

%environment
    export KERAS_MODE=TENSORFLOW
    
