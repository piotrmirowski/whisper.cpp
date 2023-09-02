#!/bin/sh

# Get device from command line or use default value.
DEVICE=$1
DEVICE="${DEVICE:=0}"
echo "Using device $DEVICE"

./talk-server -mw models/ggml-small.en.bin -c $DEVICE -vth 0.8 -vhi 0.85
