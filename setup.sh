#!/bin/bash
sudo apt-get update
sudo apt-get install -y python3-pip python3-venv libta-lib0 libta-lib0-dev build-essential
python3 -m venv venv
. venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
