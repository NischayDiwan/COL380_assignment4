#!/usr/bin/env bash
sleep 5s
nvcc test.cu -o test
nvprof ./test