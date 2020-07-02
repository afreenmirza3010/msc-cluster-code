Bootstrap: docker
From: tensorflow/tensorflow:latest-gpu-py3
Stage: build

%post
    apt update -y
    apt upgrade -y
    pip install ipython
    pip install tensorflow-gpu==2.00
    pip install http://github.com/huynhngoc/deoxys/archive/master.zip
    pip install comet-ml
    pip install scikit-image
    

    
