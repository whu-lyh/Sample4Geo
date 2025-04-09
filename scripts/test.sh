#!/usr/bin/env bash
clear
GPUS="0"

cd ../

CUDA_VISIBLE_DEVICES=${GPUS} HF_ENDPOINT=https://hf-mirror.com python eval_university.py