#!/bin/bash

export CUDA_AVAILABLE_DEVICES=1

python main.py -c configs/uadetrac_reid_fri.yml
