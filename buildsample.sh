#!/bin/bash

set -x
set -e

clang++-3.8 -fPIC -I../include -c -o sgemm.o ../samples/sgemm.c
clang++-3.8 -pie -o sgemm sgemm.o -lOpenCL -L. -lclblast
LD_LIBRARY_PATH=. ./sgemm
