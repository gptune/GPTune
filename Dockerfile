FROM ubuntu:18.04

WORKDIR /app
RUN apt-get update
RUN apt-get install git -y
RUN git clone https://github.com/gptune/GPTune
WORKDIR GPTune
RUN bash config_ubuntu_moreinstall.sh 
