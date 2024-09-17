# huggingface: https://huggingface.co/MahmoudAshraf/mms-300m-1130-forced-aligner

FROM nvidia/cuda:12.6.1-cudnn-devel-ubuntu20.04

RUN apt-get update -y && apt-get upgrade -y

# # install python3.10
# RUN DEBIAN_FRONTEND=noninteractive apt-get install -y software-properties-common
# RUN add-apt-repository -y ppa:deadsnakes/ppa
# RUN apt-get update -y
# RUN apt install -y python3.10
# RUN ln -s /usr/bin/python3.10 /usr/bin/python

# RUN DEBIAN_FRONTEND=noninteractive apt-get install -y \
#     git-core \
#     curl \
#     gcc \
#     g++

# RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python

RUN DEBIAN_FRONTEND=noninteractive apt-get install -y \
    python3 \
    python3-pip

RUN useradd -m appuser
USER appuser

RUN pip install numpy
# RUN pip install torch==2.3.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu121
# RUN python -m pip install git+https://github.com/MahmoudAshraf97/ctc-forced-aligner.git

WORKDIR /opt/app

ENTRYPOINT [ "/bin/bash" ]