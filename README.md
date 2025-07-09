# OpenCL-Wrapper
OpenCL is the most powerful programming language ever created. Yet the OpenCL C++ bindings are cumbersome and the code overhead prevents many people from getting started.
I created this lightweight OpenCL-Wrapper to greatly simplify OpenCL software development with C++ while keeping functionality and performance.

Works in Windows, Linux and Android with C++17.

Use-case example: [FluidX3D](https://github.com/ProjectPhysX/FluidX3D) builds entirely on top of this OpenCL-Wrapper.

## Getting started:
<details><summary>Install GPU Drivers and OpenCL Runtime (click to expand section)</summary>

- **Windows**
  <details><summary>GPUs</summary>

  - Download and install the [AMD](https://www.amd.com/en/support/download/drivers.html)/[Intel](https://www.intel.com/content/www/us/en/download/785597/intel-arc-iris-xe-graphics-windows.html)/[Nvidia](https://www.nvidia.com/Download/index.aspx) GPU Drivers, which contain the OpenCL Runtime.
  - Reboot.

  </details>
  <details><summary>CPUs</summary>

  - Download and install the [Intel CPU Runtime for OpenCL](https://www.intel.com/content/www/us/en/developer/articles/technical/intel-cpu-runtime-for-opencl-applications-with-sycl-support.html) (works for both AMD/Intel CPUs).
  - Reboot.

  </details>
- **Linux**
  <details><summary>AMD GPUs</summary>

  - Download and install [AMD GPU Drivers](https://www.amd.com/en/support/download/linux-drivers.html), which contain the OpenCL Runtime, with:
    ```bash
    sudo apt update && sudo apt upgrade -y
    sudo apt install -y g++ git make ocl-icd-libopencl1 ocl-icd-opencl-dev
    mkdir -p ~/amdgpu
    wget -P ~/amdgpu https://repo.radeon.com/amdgpu-install/6.4.1/ubuntu/noble/amdgpu-install_6.4.60401-1_all.deb
    sudo apt install -y ~/amdgpu/amdgpu-install*.deb
    sudo amdgpu-install -y --usecase=graphics,rocm,opencl --opencl=rocr
    sudo usermod -a -G render,video $(whoami)
    rm -r ~/amdgpu
    sudo shutdown -r now
    ```

  </details>
  <details><summary>Intel GPUs</summary>

  - Intel GPU Drivers come already installed since Linux Kernel 6.2, but they don't contain the OpenCL Runtime.
  - The the [OpenCL Runtime](https://github.com/intel/compute-runtime/releases) has to be installed separately with:
    ```bash
    sudo apt update && sudo apt upgrade -y
    sudo apt install -y g++ git make ocl-icd-libopencl1 ocl-icd-opencl-dev intel-opencl-icd
    sudo usermod -a -G render $(whoami)
    sudo shutdown -r now
    ```

  </details>
  <details><summary>Nvidia GPUs</summary>

  - Download and install [Nvidia GPU Drivers](https://www.nvidia.com/Download/index.aspx), which contain the OpenCL Runtime, with:
    ```bash
    sudo apt update && sudo apt upgrade -y
    sudo apt install -y g++ git make ocl-icd-libopencl1 ocl-icd-opencl-dev nvidia-driver-570
    sudo shutdown -r now
    ```

  </details>
  <details><summary>CPUs</summary>

  - Option 1: Download and install the [oneAPI DPC++ Compiler](https://github.com/intel/llvm/releases?q=%22oneAPI+DPC%2B%2B+Compiler+dependencies%22) and [oneTBB](https://github.com/uxlfoundation/oneTBB/releases) with:
    ```bash
    export OCLV="oclcpuexp-2025.20.6.0.04_224945_rel"
    export TBBV="oneapi-tbb-2022.2.0"
    sudo apt update && sudo apt upgrade -y
    sudo apt install -y g++ git make ocl-icd-libopencl1 ocl-icd-opencl-dev
    sudo mkdir -p ~/cpurt /opt/intel/${OCLV} /etc/OpenCL/vendors /etc/ld.so.conf.d
    sudo wget -P ~/cpurt https://github.com/intel/llvm/releases/download/2025-WW27/${OCLV}.tar.gz
    sudo wget -P ~/cpurt https://github.com/uxlfoundation/oneTBB/releases/download/v2022.2.0/${TBBV}-lin.tgz
    sudo tar -zxvf ~/cpurt/${OCLV}.tar.gz -C /opt/intel/${OCLV}
    sudo tar -zxvf ~/cpurt/${TBBV}-lin.tgz -C /opt/intel
    echo /opt/intel/${OCLV}/x64/libintelocl.so | sudo tee /etc/OpenCL/vendors/intel_expcpu.icd
    echo /opt/intel/${OCLV}/x64 | sudo tee /etc/ld.so.conf.d/libintelopenclexp.conf
    sudo ln -sf /opt/intel/${TBBV}/lib/intel64/gcc4.8/libtbb.so /opt/intel/${OCLV}/x64
    sudo ln -sf /opt/intel/${TBBV}/lib/intel64/gcc4.8/libtbbmalloc.so /opt/intel/${OCLV}/x64
    sudo ln -sf /opt/intel/${TBBV}/lib/intel64/gcc4.8/libtbb.so.12 /opt/intel/${OCLV}/x64
    sudo ln -sf /opt/intel/${TBBV}/lib/intel64/gcc4.8/libtbbmalloc.so.2 /opt/intel/${OCLV}/x64
    sudo ldconfig -f /etc/ld.so.conf.d/libintelopenclexp.conf
    sudo rm -r ~/cpurt
    ```
  - Option 2: Download and install [PoCL](https://portablecl.org/) with:
    ```bash
    sudo apt update && sudo apt upgrade -y
    sudo apt install -y g++ git make ocl-icd-libopencl1 ocl-icd-opencl-dev pocl-opencl-icd
    ```
  </details>

- **Android**
  <details><summary>ARM GPUs</summary>

  - Download the [Termux `.apk`](https://github.com/termux/termux-app/releases) and install it.
  - In the Termux app, run:
    ```bash
    apt update && apt upgrade -y
    apt install -y clang git make
    ```

  </details>

</details>

&#9656;[Download](https://github.com/ProjectPhysX/OpenCL-Wrapper/archive/refs/heads/master.zip)+unzip the source code or `git clone https://github.com/ProjectPhysX/OpenCL-Wrapper.git`

<details><summary>Compiling on Windows (click to expand section)</summary>

- Download and install [Visual Studio Community](https://visualstudio.microsoft.com/de/vs/community/). In Visual Studio Installer, add:
  - Desktop development with C++
  - MSVC v142
  - Windows 10 SDK
- Open [`OpenCL-Wrapper.sln`](OpenCL-Wrapper.sln) in [Visual Studio Community](https://visualstudio.microsoft.com/de/vs/community/).
- Compile and run by clicking the <kbd>► Local Windows Debugger</kbd> button.

</details>
<details><summary>Compiling on Linux / macOS / Android (click to expand section)</summary>

- Compile and run with:
  ```bash
  chmod +x make.sh
  ./make.sh
  ```
- Compiling requires [`g++`](https://gcc.gnu.org/) with `C++17`, which is supported since version `8` (check with `g++ --version`).
- Operating system (Linux/macOS/Android) is detected automatically.

</details>

## Key simplifications:
1. select a `Device` with 1 line
   - automatically select fastest device / device with most memory / device with specified ID from a list of all devices
   - easily get device information (performance in TFLOPs/s, amount of memory and cache, FP64/FP16 capabilities, etc.)
   - automatic OpenCL C code compilation when creating the Device object
     - automatically enable FP64/FP16 capabilities in OpenCL C code
     - automatically print log to console if there are compile errors
     - easy option to generate PTX assembly for Nvidia GPUs and save that in a `.ptx` file
   - <details><summary>contains all device-specific workarounds/patches to make OpenCL fully cross-compatible</summary>

     - AMD
       - fix for wrong device name reporting on AMD GPUs
       - fix for wrong reporting of dual-CU count as CU count on AMD RDNA+ GPUs
       - fix for maximum buffer allocation size limit for AMD GPUs
     - Intel
       - enable >4GB single buffer VRAM allocations on Intel Arc GPUs
       - fix for wrong VRAM capacity reporting on Intel Arc GPUs
       - fix for maximum buffer allocation size limit in Intel CPU Runtime for OpenCL
       - fix for false dp4a reporting on Intel
     - Nvidia
       - enable basic FP16 support on Nvidia Pascal and newer GPUs with driver 520 or newer
     - other
       - enable FP64, FP16 and INT64 atomics support on supported devices
       - fix for unreliable OpenCL C version reporting
       - always compile for latest supported OpenCL C standard
       - fix for terrible `fma` performance on ARM GPUs

     </details>
2. create a `Memory` object with 1 line
   - one object for both host and device memory
   - easy host <-> device memory transfer (also for 1D/2D/3D grid domains)
   - easy handling of multi-dimensional vectors
   - can also be used to only allocate memory on host or only allocate memory on device
   - automatically tracks total global memory usage of device when allocating/deleting memory
   - automatically uses zero-copy buffers on CPUs/iGPUs
3. create a `Kernel` with 1 line
   - Memory objects and constants are linked to OpenCL C kernel parameters during Kernel creation
   - a list of Memory objects and constants can be added to Kernel parameters in one line (`add_parameters(...)`)
   - Kernel parameters can be edited (`set_parameters(...)`)
   - easy Kernel execution: `kernel.run();`
   - Kernel function calls can be daisy chained, for example: `kernel.set_parameters(3u, time).run();`
   - failsafe: you'll get an error message if kernel parameters mismatch between C++ and OpenCL code
4. OpenCL C code is embedded into C++
   - syntax highlighting in the code editor is retained
   - notes / peculiarities of this workaround:
     - the `#define R(...) string(" "#__VA_ARGS__" ")` stringification macro converts its arguments to string literals; `'\n'` is converted to `' '` in the process
     - these string literals cannot be arbitrarily long, so interrupt them periodically with `)+R(`
     - to use unbalanced round brackets `'('`/`')'`, exit the `R(...)` macro and insert a string literal manually: `)+"void function("+R(` and `)+") {"+R(`
     - to use preprocessor switch macros, exit the `R(...)` macro and insert a string literal manually: `)+"#define TEST"+R(` and `)+"#endif"+R( // TEST`
     - preprocessor replacement macros (for example `#define VARIABLE 42`) don't work; hand these to the `Device` constructor directly instead

## No need to:
- have code overhead for selecting a platform/device, passing the OpenCL C code, etc.
- keep track of length and data type for buffers
- have duplicate code for host and device buffers
- keep track of total global memory usage
- keep track of global/local range for kernels
- bother with Queue, Context, Source, Program
- load a `.cl` file at runtime
- bother with device-specific workarounds/patches

## Example (OpenCL vector addition)
### main.cpp
```c
#include "opencl.hpp"

int main() {
	Device device(select_device_with_most_flops()); // compile OpenCL C code for the fastest available device

	const uint N = 1024u; // size of vectors
	Memory<float> A(device, N); // allocate memory on both host and device
	Memory<float> B(device, N);
	Memory<float> C(device, N);

	Kernel add_kernel(device, N, "add_kernel", A, B, C); // kernel that runs on the device

	for(uint n=0u; n<N; n++) {
		A[n] = 3.0f; // initialize memory
		B[n] = 2.0f;
		C[n] = 1.0f;
	}

	print_info("Value before kernel execution: C[0] = "+to_string(C[0]));

	A.write_to_device(); // copy data from host memory to device memory
	B.write_to_device();
	add_kernel.run(); // run add_kernel on the device
	C.read_from_device(); // copy data from device memory to host memory

	print_info("Value after kernel execution: C[0] = "+to_string(C[0]));

	wait();
	return 0;
}
```

### kernel.cpp
```c
#include "kernel.hpp" // note: unbalanced round brackets () are not allowed and string literals can't be arbitrarily long, so periodically interrupt with )+R(
string opencl_c_container() { return R( // ########################## begin of OpenCL C code ####################################################################



kernel void add_kernel(global float* A, global float* B, global float* C) { // equivalent to "for(uint n=0u; n<N; n++) {", but executed in parallel
	const uint n = get_global_id(0);
	C[n] = A[n]+B[n];
}



);} // ############################################################### end of OpenCL C code #####################################################################
```

### For comparison, the very same OpenCL vector addition example looks like this when directly using the OpenCL C++ bindings:
```c
#define CL_HPP_MINIMUM_OPENCL_VERSION 100
#define CL_HPP_TARGET_OPENCL_VERSION 300
#include <CL/opencl.hpp>
#include "utilities.hpp"

#define WORKGROUP_SIZE 64

int main() {

	// 1. select device

	vector<cl::Device> cl_devices; // get all devices of all platforms
	{
		vector<cl::Platform> cl_platforms; // get all platforms (drivers)
		cl::Platform::get(&cl_platforms);
		for(uint i=0u; i<(uint)cl_platforms.size(); i++) {
			vector<cl::Device> cl_devices_available;
			cl_platforms[i].getDevices(CL_DEVICE_TYPE_ALL, &cl_devices_available);
			for(uint j=0u; j<(uint)cl_devices_available.size(); j++) {
				cl_devices.push_back(cl_devices_available[j]);
			}
		}
	}
	cl::Device cl_device; // select fastest available device
	{
		float best_value = 0.0f;
		uint best_i = 0u; // index of fastest device
		for(uint i=0u; i<(uint)cl_devices.size(); i++) { // find device with highest (estimated) floating point performance
			const string name = trim(cl_device.getInfo<CL_DEVICE_NAME>()); // device name
			const string vendor = trim(cl_device.getInfo<CL_DEVICE_VENDOR>()); // device vendor
			const uint compute_units = (uint)cl_device.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>(); // compute units (CUs) can contain multiple cores depending on the microarchitecture
			const uint clock_frequency = (uint)cl_device.getInfo<CL_DEVICE_MAX_CLOCK_FREQUENCY>(); // in MHz
			const bool is_gpu = cl_device.getInfo<CL_DEVICE_TYPE>()==CL_DEVICE_TYPE_GPU;
			const int vendor_id = (int)cl_device.getInfo<CL_DEVICE_VENDOR_ID>(); // AMD=0x1002, Intel=0x8086, Nvidia=0x10DE, Apple=0x1027F00
			float cores_per_cu = 1.0f;
			if(vendor_id==0x1002) { // AMD GPU/CPU
				const bool amd_128_cores_per_dualcu = contains(to_lower(name), "gfx10"); // identify RDNA/RDNA2 GPUs where dual CUs are reported
				const bool amd_256_cores_per_dualcu = contains(to_lower(name), "gfx11"); // identify RDNA3 GPUs where dual CUs are reported
				cores_per_cu = is_gpu ? (amd_256_cores_per_dualcu ? 256.0f : amd_128_cores_per_dualcu ? 128.0f : 64.0f) : 0.5f; // 64 cores/CU (GCN, CDNA), 128 cores/dualCU (RDNA, RDNA2), 256 cores/dualCU (RDNA3), 1/2 core/CU (CPUs)
			} else if(vendor_id==0x8086) { // Intel GPU/CPU
				const bool intel_16_cores_per_cu = contains_any(to_lower(name), {"gpu max", "140v", "130v", "b580", "b570"}); // identify PVC/Xe2 GPUs
				cores_per_cu = is_gpu ? (intel_16_cores_per_cu ? 16.0f : 8.0f) : 0.5f; // Intel GPUs have 16 cores/CU (PVC) or 8 cores/CU (integrated/Arc), Intel CPUs (with HT) have 1/2 core/CU
			} else if(vendor_id==0x10DE||vendor_id==0x13B5) { // Nvidia GPU/CPU
				const uint nvidia_compute_capability = 10u*(uint)cl_device.getInfo<CL_DEVICE_COMPUTE_CAPABILITY_MAJOR_NV>()+(uint)cl_device.getInfo<CL_DEVICE_COMPUTE_CAPABILITY_MINOR_NV>();
				const bool nvidia__32_cores_per_cu = (nvidia_compute_capability <30); // identify Fermi GPUs
				const bool nvidia_192_cores_per_cu = (nvidia_compute_capability>=30&&nvidia_compute_capability<50); // identify Kepler GPUs
				const bool nvidia__64_cores_per_cu = (nvidia_compute_capability>=70&&nvidia_compute_capability<80)||contains_any(to_lower(name), {"p100", "a100", "a30"}); // identify Volta, Turing, P100, A100, A30
				cores_per_cu = is_gpu ? (nvidia__32_cores_per_cu ? 32.0f : nvidia_192_cores_per_cu ? 192.0f : nvidia__64_cores_per_cu ? 64.0f : 128.0f) : 1.0f; // 32 (Fermi), 192 (Kepler), 64 (Volta, Turing, P100, A100, A30), 128 (Maxwell, Pascal, Ampere, Hopper, Ada, Blackwell) or 1 (CPUs)
			} else if(vendor_id==0x1027F00) { // Apple iGPU
				cores_per_cu = 128.0f; // Apple ARM GPUs usually have 128 cores/CU
			} else if(vendor_id==0x1022||vendor_id==0x10006||vendor_id==0x6C636F70) { // x86 CPUs with PoCL runtime
				cores_per_cu = 0.5f; // CPUs typically have 1/2 cores/CU due to SMT/hyperthreading
			} else if(contains(to_lower(vendor), "arm")) { // ARM
				cores_per_cu = is_gpu ? 8.0f : 1.0f; // ARM GPUs usually have 8 cores/CU, ARM CPUs have 1 core/CU
			}
			const uint ipc = is_gpu ? 2u : 32u; // IPC (instructions per cycle) is 2 for GPUs and 32 for most modern CPUs
			const uint cores = to_uint((float)compute_units*cores_per_cu); // for CPUs, compute_units is the number of threads (twice the number of cores with hyperthreading)
			const float tflops = 1E-6f*(float)cores*(float)ipc*(float)clock_frequency; // estimated device floating point performance in TeraFLOPs/s
			if(tflops>best_value) {
				best_value = tflops;
				best_i = i;
			}
		}
		const string name = trim(cl_devices[best_i].getInfo<CL_DEVICE_NAME>()); // device name
		cl_device = cl_devices[best_i];
		print_info(name); // print device name
	}

	// 2. embed OpenCL C code (raw string literal breaks syntax highlighting)

	string opencl_c_code = R"(
		kernel void add_kernel(global float* A, global float* B, global float* C) { // equivalent to "for(uint n=0u; n<N; n++) {", but executed in parallel
			const uint n = get_global_id(0);
			C[n] = A[n]+B[n];
		}
	)";

	// 3. compile OpenCL C code

	cl::Context cl_context;
	cl::Program cl_program;
	cl::CommandQueue cl_queue;
	{
		cl_context = cl::Context(cl_device);
		cl_queue = cl::CommandQueue(cl_context, cl_device);
		cl::CommandQueue cl_queue(cl_context, cl_device); // queue to push commands for the device
		cl::Program::Sources cl_source;
		cl_source.push_back({ opencl_c_code.c_str(), opencl_c_code.length() });
		cl_program = cl::Program(cl_context, cl_source);
		int error = cl_program.build({ cl_device }, "-cl-finite-math-only -cl-no-signed-zeros -cl-mad-enable -w"); // compile OpenCL C code, disable warnings
		if(error) print_warning(cl_program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(cl_device)); // print build log
		if(error) print_error("OpenCL C code compilation failed.");
		else print_info("OpenCL C code successfully compiled.");
	}

	// 4. allocate memory on host and device

	const uint N = 1024u;
	float* host_A;
	float* host_B;
	float* host_C;
	cl::Buffer device_A;
	cl::Buffer device_B;
	cl::Buffer device_C;
	{
		host_A = new float[N];
		host_B = new float[N];
		host_C = new float[N];
		for(uint i=0u; i<N; i++) {
			host_A[i] = 0.0f; // zero all buffers
			host_B[i] = 0.0f;
			host_C[i] = 0.0f;
		}
		int error = 0;
		device_A = cl::Buffer(cl_context, CL_MEM_READ_WRITE, N*sizeof(float), nullptr, &error);
		if(error) print_error("OpenCL Buffer allocation failed with error code "+to_string(error)+".");
		device_B = cl::Buffer(cl_context, CL_MEM_READ_WRITE, N*sizeof(float), nullptr, &error);
		if(error) print_error("OpenCL Buffer allocation failed with error code "+to_string(error)+".");
		device_C = cl::Buffer(cl_context, CL_MEM_READ_WRITE, N*sizeof(float), nullptr, &error);
		if(error) print_error("OpenCL Buffer allocation failed with error code "+to_string(error)+".");
		cl_queue.enqueueWriteBuffer(device_A, true, 0u, N*sizeof(float), (void*)host_A); // have to keep track of buffer range and buffer data type
		cl_queue.enqueueWriteBuffer(device_B, true, 0u, N*sizeof(float), (void*)host_B);
		cl_queue.enqueueWriteBuffer(device_C, true, 0u, N*sizeof(float), (void*)host_C);
	}

	// 5. create Kernel object and link input parameters

	cl::NDRange cl_range_global, cl_range_local;
	cl::Kernel cl_kernel;
	{
		cl_kernel = cl::Kernel(cl_program, "add_kernel");
		cl_kernel.setArg(0, device_A);
		cl_kernel.setArg(1, device_B);
		cl_kernel.setArg(2, device_C);
		cl_range_local = cl::NDRange(WORKGROUP_SIZE);
		cl_range_global = cl::NDRange(((N+WORKGROUP_SIZE-1)/WORKGROUP_SIZE)*WORKGROUP_SIZE); // make global range a multiple of local range
	}

	// 6. finally run the actual program

	{
		for(uint i=0u; i<N; i++) {
			host_A[i] = 3.0f; // initialize buffers on host
			host_B[i] = 2.0f;
			host_C[i] = 1.0f;
		}

		print_info("Value before kernel execution: C[0] = "+to_string(host_C[0]));

		cl_queue.enqueueWriteBuffer(device_A, true, 0u, N*sizeof(float), (void*)host_A); // copy A and B to device
		cl_queue.enqueueWriteBuffer(device_B, true, 0u, N*sizeof(float), (void*)host_B); // have to keep track of buffer range and buffer data type
		cl_queue.enqueueNDRangeKernel(cl_kernel, cl::NullRange, cl_range_global, cl_range_local); // have to keep track of kernel ranges
		cl_queue.finish(); // don't forget to finish the queue
		cl_queue.enqueueReadBuffer(device_C, true, 0u, N*sizeof(float), (void*)host_C);

		print_info("Value after kernel execution: C[0] = "+to_string(host_C[0]));
	}

	wait();
	return 0;
}
```