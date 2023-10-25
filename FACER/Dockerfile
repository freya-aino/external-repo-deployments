FROM ubuntu:20.04

WORKDIR /root

RUN apt-get update -y
RUN apt-get upgrade -y

RUN apt-get install python3 python3-pip -y
RUN DEBIAN_FRONTEND=noninteractive apt-get install ffmpeg libsm6 libxext6 -y
RUN apt install git -y

RUN pip3 install git+https://github.com/FacePerceiver/facer.git@main
RUN pip3 install timm

COPY ./process.py ./process.py
RUN pip3 install imagezmq
ENV ADDRESS="tcp://192.168.2.102:6666"

ENTRYPOINT [ "/bin/bash" ]

