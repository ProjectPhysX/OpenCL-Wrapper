#include "opencl.hpp"

int main() {
	const Device device(select_device_with_most_flops()); // compile OpenCL C code for the fastest available device

	const uint N = 1024u; // size of vectors

	Memory<float> A(device, N); // allocate memory on both host and device
	Memory<float> B(device, N);
	Memory<float> C(device, N);

	for(uint i=0u; i<N; i++) {
		A[i] = 3.0f; // initialize memory
		B[i] = 2.0f;
		C[i] = 1.0f;
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