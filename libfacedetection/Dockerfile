FROM nvidia/cuda:11.8.0-devel-ubuntu20.04

SHELL [ "/bin/bash", "-c"]
RUN ln -snf /usr/share/zoneinfo/$CONTAINER_TIMEZONE /etc/localtime && echo $CONTAINER_TIMEZONE > /etc/timezone

RUN apt-get update -y
RUN apt-get upgrade -y

RUN apt-get install software-properties-common -y
RUN add-apt-repository ppa:deadsnakes/ppa -y
RUN apt-get update -y
RUN apt-get install wget -y
RUN apt-get install python3-pip -y

# WORKDIR /opt/
# # RUN wget https://repo.anaconda.com/archive/Anaconda3-2023.07-2-Linux-x86_64.sh
# # RUN chmod u+x Anaconda3-2023.07-2-Linux-x86_64.sh
# # RUN ./Anaconda3-2023.07-2-Linux-x86_64.sh -b

# # ENV PATH /root/anaconda3/bin:$PATH

# # RUN conda init bash

# # SETUP CONDA ENVIRONMENT
# # RUN /opt/conda/bin/activate
# # RUN conda create -n vedadet python=3.10 -y
# # SHELL [ "conda", "run", "-n", "vedadet", "/bin/bash", "-c" ]
# # RUN conda install pytorch=1.7.3 -c pytorch -c
# # RUN conda install cuda -c nvidia/label/cuda-11.8.0 -y
# # RUN conda install cudatoolkit=11.8 -c nvidia -y

RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# NNCV INSTALL
RUN pip install -U openmim
RUN mim install mmcv mmengine mmdet

# SETUP VEDADET PROJECT
COPY . /app/
WORKDIR /app/libfacedetection
RUN python3 setup.py develop
RUN pip install -r requirements.txt


ENTRYPOINT [ "./../build_and_run.sh" ] 