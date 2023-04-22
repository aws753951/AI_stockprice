FROM jupyter/datascience-notebook

USER root

RUN apt-get update -y &&\
  apt-get upgrade -y &&\
  apt-get install -y vim wget curl python3-pip

RUN pip3 install ipykernel torch matplotlib numpy pandas

COPY Dockerfile .