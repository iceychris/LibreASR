#!/bin/bash

PL=125

# enable persistence mode
sudo nvidia-smi -i 0 -pm ENABLED
sudo nvidia-smi -i 1 -pm ENABLED

# set wattage / power limit
sudo nvidia-smi -i 0 -pl ${PL}
sudo nvidia-smi -i 1 -pl ${PL}
