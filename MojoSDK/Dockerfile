FROM ubuntu:20.04

USER root
WORKDIR /opt/app

ARG port=8888
ARG password=""


RUN ln -snf /usr/share/zoneinfo/$CONTAINER_TIMEZONE /etc/localtime && echo $CONTAINER_TIMEZONE > /etc/timezone

RUN apt-get update
RUN apt-get upgrade
RUN DEBIAN_FRONTEND=noninteractive \
    ${DEFAULT_TZ} \
    apt-get install -y \
    curl \
    apt-transport-https \
    python3.10 \
    pip

RUN python3 -m pip install \
    jupyterlab \
    ipykernel \
    matplotlib \
    ipywidgets

RUN curl -s https://get.modular.com | sh -
RUN modular install mojo

ENV MODULAR_HOME=/root/.modular
ENV PATH="${PATH}:${MODULAR_HOME}/pkg/packages.modular.com_mojo/bin"

CMD ["jupyter", "lab", "--ip='*'", "--port=${port}", "--no-browser", "--allow-root", "--NotebookApp.token=''", "--NotebookApp.password='${password}'"]