#!/usr/bin/env bash

mkdir -p bin # create directory for executable
rm -f bin/OpenCL-Wrapper # prevent execution of old version if compiling fails

case "$(uname -a)" in # automatically detect operating system
	 Darwin*) g++ src/*.cpp -o bin/OpenCL-Wrapper -std=c++17 -pthread -O -Wno-comment -I./src/OpenCL/include -L./src/OpenCL/lib -lOpenCL     ;; # macOS
	*Android) g++ src/*.cpp -o bin/OpenCL-Wrapper -std=c++17 -pthread -O -Wno-comment -I./src/OpenCL/include -framework OpenCL               ;; # Android
	*       ) g++ src/*.cpp -o bin/OpenCL-Wrapper -std=c++17 -pthread -O -Wno-comment -I./src/OpenCL/include -L/system/vendor/lib64 -lOpenCL ;; # Linux
esac

if [[ $? == 0 ]]; then bin/OpenCL-Wrapper "$@"; fi # run executable only if last compilation was successful
