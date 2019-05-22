#!/bin/bash

sudo apt-get -f install
sudo apt install python3-pip
pip3 install --upgrade pip
pip3 install -U pandas --user
pip3 install seaborn --user
pip3 install matplotlib --user
pip3 install -U scikit-learn --user
