#!/bin/bash

sudo rm /var/lib/apt/lists/lock
sudo rm /var/lib/dpkg/lock
sudo apt-get update
sudo apt-get upgrade
