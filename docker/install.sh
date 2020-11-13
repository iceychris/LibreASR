#!/bin/bash

###
# Install script for asr-pytorch.
# Assumes that python3, pip3, torch and torchvision are already installed.
###


###
# apt
###

apt-get update
apt-get install -y gcc g++ sox libsox-dev libsox-fmt-all vim make \
    python3-dev portaudio19-dev alsa-utils


###
# python requirements
###

# wrap pip command
alias pip3="python3 -m pip"

# install reqs
pip3 install -r requirements.inference.txt
