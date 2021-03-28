# FROM ubuntu:16.04
# FROM debian:stable
FROM ubuntu:18.04

WORKDIR /app
RUN apt-get update
RUN apt-get install git -y
RUN git clone https://github.com/gptune/GPTune.git
WORKDIR GPTune
RUN git fetch
RUN bash config_cleanlinux.sh 
