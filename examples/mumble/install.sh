#!/bin/sh

# update repos
apt-get update

# install opus stuff
apt-get install -y libopus0 opus-tools

# install python deps
pip install pymumble opuslib