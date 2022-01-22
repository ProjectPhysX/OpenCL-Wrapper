mkdir -p bin
g++ ./src/*.cpp -o ./bin/OpenCL-Wrapper.exe -std=c++17 -pthread -w -I./src/OpenCL/include -L./src/OpenCL/lib -lOpenCL
#g++ ./src/*.cpp -o ./bin/OpenCL-Wrapper.exe -std=c++17 -pthread -w -I./src/OpenCL/include -L/system/vendor/lib64 -lOpenCL
./bin/OpenCL-Wrapper.exe
