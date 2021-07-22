#!/bin/bash

# vars
F="libtorch-cxx11-abi-shared-with-deps-1.9.0+cpu.zip"
LINK="https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-1.9.0%2Bcpu.zip"

# install prereqs
apt-get update && apt-get install -y wget unzip cmake

# download and extract libtorch to ./
wget ${LINK}
unzip ${F}
rm ${F}
