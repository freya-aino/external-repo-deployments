# huggingface: https://huggingface.co/MahmoudAshraf/mms-300m-1130-forced-aligner
# All Credit for the Implementation of the ctc-forced-aligner goes to MahmoudAshraf97

FROM nvidia/cuda:12.6.1-cudnn-devel-ubuntu20.04

RUN apt-get update -y

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



RUN apt-get update -y

RUN DEBIAN_FRONTEND=noninteractive \
    apt-get install -y \
    python3.9 \
    python3-pip

RUN apt-get install -y \
    git-core \
    sudo


RUN useradd -m -G sudo appuser
USER appuser
WORKDIR /opt/app


RUN git clone https://github.com/MahmoudAshraf97/ctc-forced-aligner.git
# RUN pip install torch==2.3.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu121

RUN cd /opt/app/ctc-forced-aligner; python3.9 -m pip install -e .[dev]


COPY . .
ENTRYPOINT [ "/bin/bash" ]