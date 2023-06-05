#!/bin/sh
if [ ! -d "buildgcc/nosan" ]; then
    CXX=g++ cmake -S . -B buildgcc/nosan/ -DCMAKE_BUILD_TYPE=Debug
fi
if [ ! -d "buildgcc/test" ]; then
    CXX=g++ cmake -S . -B buildgcc/test/ -DCMAKE_BUILD_TYPE=Debug -DUSE_SANITIZER='Address;Undefined'
fi
if [ ! -d "buildclang/nosan" ]; then
    CXX=clang++ cmake -S . -B buildclang/nosan/ -DCMAKE_BUILD_TYPE=Debug
fi
if [ ! -d "buildclang/test" ]; then
    CXX=clang++ cmake -S . -B buildclang/test/ -DCMAKE_BUILD_TYPE=Debug -DUSE_SANITIZER='Address;Undefined'
fi
time cmake --build buildgcc/nosan/ 
time cmake --build buildgcc/nosan/ --target test
time cmake --build buildclang/nosan/ 
time cmake --build buildclang/nosan/ --target test
time cmake --build buildgcc/test/ 
time cmake --build buildgcc/test/ --target test
time cmake --build buildclang/test/ 
time cmake --build buildclang/test/ --target test

