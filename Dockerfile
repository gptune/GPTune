FROM ubuntu:16.04
# FROM ubuntu:18.04

WORKDIR /app
RUN apt-get update
RUN apt-get install git -y
RUN git clone https://github.com/gptune/GPTune.git
WORKDIR GPTune
RUN bash config_ubuntu_moreinstall.sh 
