#!/bin/bash
# Remove all existing OpenCV installations
pip uninstall opencv-python opencv-contrib-python opencv-python-headless -y

# Install numpy first with correct version
pip install numpy==1.21.6 --no-cache-dir

# Install requirements without dependencies first
pip install --no-deps -r requirements.txt

# Now install just the required dependencies
pip install absl-py attrs flatbuffers matplotlib sounddevice