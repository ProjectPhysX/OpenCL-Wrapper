# OpenCL-Wrapper
OpenCL is the most powerful programming language ever created. Yet the OpenCL C++ bindings are very cumbersome and the code overhead prevents many people from getting started.
I created this lightweight OpenCL-Wrapper to greatly simplify OpenCL software development with C++ while keeping functionality and performance.

Works in both Windows and Linux with C++17.

## Key simplifications:
1. select a `Device` with 1 line
   - automatically select fastest device / device with most memory / device with specified ID from a list of all devices
   - easily get device information (performance in TFLOPs/s, amount of memory and cache, FP64/FP16 capabilities, etc.)
   - automatic OpenCL C code compilation when creating the Device object
     - automatically enable FP64/FP16 capabilities in OpenCL C code
     - automatically print log to console if there are compile errors
     - easy option to generate PTX assembly and save that in a `.ptx` file
2. create a `Memory` object with 1 line
   - one object for both host and device memory
   - easy host <-> device memory transfer
   - easy handling of multi-dimensional vectors
   - can also be used to only allocate memory on host or only allocate memory on device
3. create a `Kernel` with 1 line
   - memory objects are linked to OpenCL C kernel parameters during kernel creation
   - easy kernel execution
4. OpenCL C code is embedded into C++
   - syntax highlighting in the code editor is retained

## No need to:
- have code overhead for selecting a platform/device, passing the OpenCL C code, etc.
- keep track of global/local ranges for buffers and kernels
- have duplicate code for host and device buffers
- bother with Queue, Context, Source, Program
- load a `.cl` file at runtime

## Example (OpenCL vector addition)
### main.cpp
```c
#include "opencl.hpp"

int main() {
	const Device device(select_device_with_most_flops()); // compile OpenCL C code for the fastest available device

	const uint N = 1024u; // size of vectors

	Memory<float> A(device, N); // allocate memory on both host and device
	Memory<float> B(device, N);
	Memory<float> C(device, N);

	for(uint n=0u; n<N; n++) {
		A[n] = 3.0f; // initialize memory
		B[n] = 2.0f;
		C[n] = 1.0f;
	}

	const Kernel add_kernel(device, N, "add_kernel", A, B, C); // kernel that runs on the device

	print_info("Value before kernel execution: C[0] = "+to_string(C[0]));

	A.write_to_device(); // copy data from host memory to device memory
	B.write_to_device(); // copy data from host memory to device memory
	add_kernel.run(); // run add_kernel on the device
	C.read_from_device(); // copy data from device memory to host memory

	print_info("Value after kernel execution: C[0] = "+to_string(C[0]));

	wait();
	return 0;
}
```

### kernel.cpp
```c
#include "kernel.hpp" // note: string literals can't be arbitrarily long, so periodically interrupt with )+R(
string opencl_c_container() { return R( // ########################## begin of OpenCL C code ####################################################################



kernel void add_kernel(global float* A, global float* B, global float* C) { // equivalent to "for(int n=0; n<N; n++) {", but executed in parallel
	const uint n = get_global_id(0);
	C[n] = A[n]+B[n];
}



);} // ############################################################### end of OpenCL C code #####################################################################
```
