#!/usr/bin/env bash

case "$(uname -a)" in # automatically detect operating system
	 Darwin*) target=macOS   ;;
	*Android) target=Android ;;
	*       ) target=Linux   ;;
esac

#target=Linux   # manually set to compile on Linux
#target=macOS   # manually set to compile on macOS
#target=Android # manually set to compile on Android

echo -e "\033[92mInfo\033[0m: Compiling for Operating System: "${target}
mkdir -p bin # create directory for executable
rm -f ./bin/OpenCL-Wrapper # prevent execution of old version if compiling fails

case "${target}" in
	Linux    ) g++ src/*.cpp -o bin/OpenCL-Wrapper -std=c++17 -pthread -O -Wno-comment -I./src/OpenCL/include -L./src/OpenCL/lib -lOpenCL     ;;
	macOS    ) g++ src/*.cpp -o bin/OpenCL-Wrapper -std=c++17 -pthread -O -Wno-comment -I./src/OpenCL/include -framework OpenCL               ;;
	Android  ) g++ src/*.cpp -o bin/OpenCL-Wrapper -std=c++17 -pthread -O -Wno-comment -I./src/OpenCL/include -L/system/vendor/lib64 -lOpenCL ;;
esac

if [[ $? == 0 ]]; then bin/OpenCL-Wrapper "$@"; fi # run executable only if last compilation was successful
