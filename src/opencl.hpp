#pragma once

#define THREAD_BLOCK_SIZE 32
//#define PTX
//#define LOG
//#define USE_OPENCL_1_1

#ifdef USE_OPENCL_1_1
#define CL_USE_DEPRECATED_OPENCL_1_1_APIS
#endif // USE_OPENCL_1_1
#define to_string to_string_old // use own to_string methods
#include <CL/cl.hpp> // old version (OpenCL 1.0, 1.1, 1.2)
#undef to_string // use own to_string methods
#include "utilities.hpp"

struct Device_Info {
	cl::Device cl_device;
	string name, vendor; // device name, vendor
	string driver_version, opencl_c_version; // device driver version, OpenCL C version
	uint memory=0u; // global memory in MB
	uint memory_used=0u; // track global memory usage in MB
	uint global_cache=0u, local_cache=0u; // global cache in KB, local cache in KB
	uint max_global_buffer=0u, max_constant_buffer=0u; // maximum global buffer size in MB, maximum constant buffer size in KB
	uint compute_units=0u; // compute units (CUs) can contain multiple cores depending on the microarchitecture
	uint clock_frequency=0u; // in MHz
	bool is_cpu=false, is_gpu=false;
	uint is_fp64_capable=0u, is_fp32_capable=0u, is_fp16_capable=0u, is_int64_capable=0u, is_int32_capable=0u, is_int16_capable=0u, is_int8_capable=0u;
	uint cores=0u; // for CPUs, compute_units is the number of threads (twice the number of cores with hyperthreading)
	float tflops=0.0f; // estimated device floating point performance in TeraFLOPs/s
	inline Device_Info() = default;
	inline Device_Info(const cl::Device& cl_device) {
		this->cl_device = cl_device; // see https://www.khronos.org/registry/OpenCL/sdk/1.2/docs/man/xhtml/clGetDeviceInfo.html
		name = trim(cl_device.getInfo<CL_DEVICE_NAME>()); // device name
		vendor = trim(cl_device.getInfo<CL_DEVICE_VENDOR>()); // device vendor
		driver_version = trim(cl_device.getInfo<CL_DRIVER_VERSION>()); // device driver version
		opencl_c_version = trim(cl_device.getInfo<CL_DEVICE_OPENCL_C_VERSION>()); // device OpenCL C version
		memory = (uint)(cl_device.getInfo<CL_DEVICE_GLOBAL_MEM_SIZE>()/1048576ull); // global memory in MB
		global_cache = (uint)(cl_device.getInfo<CL_DEVICE_GLOBAL_MEM_CACHE_SIZE>()/1024ull); // global cache in KB
		local_cache = (uint)(cl_device.getInfo<CL_DEVICE_LOCAL_MEM_SIZE>()/1024ull); // local cache in KB
		max_global_buffer = (uint)(cl_device.getInfo<CL_DEVICE_MAX_MEM_ALLOC_SIZE>()/1048576ull); // maximum global buffer size in MB
		max_constant_buffer = (uint)(cl_device.getInfo<CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE>()/1024ull); // maximum constant buffer size in KB
		compute_units = (uint)cl_device.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>(); // compute units (CUs) can contain multiple cores depending on the microarchitecture
		clock_frequency = (uint)cl_device.getInfo<CL_DEVICE_MAX_CLOCK_FREQUENCY>(); // in MHz
		is_fp64_capable = (uint)cl_device.getInfo<CL_DEVICE_NATIVE_VECTOR_WIDTH_DOUBLE>();
		is_fp32_capable = (uint)cl_device.getInfo<CL_DEVICE_NATIVE_VECTOR_WIDTH_FLOAT>();
		is_fp16_capable = (uint)cl_device.getInfo<CL_DEVICE_NATIVE_VECTOR_WIDTH_HALF>();
		is_int64_capable = (uint)cl_device.getInfo<CL_DEVICE_NATIVE_VECTOR_WIDTH_LONG>();
		is_int32_capable = (uint)cl_device.getInfo<CL_DEVICE_NATIVE_VECTOR_WIDTH_INT>();
		is_int16_capable = (uint)cl_device.getInfo<CL_DEVICE_NATIVE_VECTOR_WIDTH_SHORT>();
		is_int8_capable = (uint)cl_device.getInfo<CL_DEVICE_NATIVE_VECTOR_WIDTH_CHAR>();
		is_cpu = cl_device.getInfo<CL_DEVICE_TYPE>()==CL_DEVICE_TYPE_CPU;
		is_gpu = cl_device.getInfo<CL_DEVICE_TYPE>()==CL_DEVICE_TYPE_GPU;
		const uint ipc = is_gpu?2u:32u; // IPC (instructions per cycle) is 2 for GPUs and 32 for most modern CPUs
		const float nvidia = (float)(contains(to_lower(vendor), "nvidia"))*(clock_frequency<1000u?192.0f:compute_units<=30u?128.0f:64.0f); // Nvidia GPUs have 192 cores/CU (Kepler), 128 cores/CU (Maxwell, Pascal, Ampere) or 64 cores/CU (P100, Volta, Turing, A100)
		const float amd = (float)(contains(to_lower(vendor), "amd")||contains(to_lower(vendor), "advanced"))*(is_gpu?64.0f:0.5f); // AMD GCN GPUs usually have 64 cores/CU, AMD CPUs have 1/2 core/CU
		const float intel = (float)(contains(to_lower(vendor), "intel"))*(is_gpu?8.0f:0.5f); // Intel integrated GPUs usually have 8 cores/CU, Intel CPUs have 1/2 core/CU
		const float arm = (float)(contains(to_lower(vendor), "arm"))*(is_gpu?8.0f:1.0f); // ARM GPUs usually have 8 cores/CU, ARM CPUs have 1 core/CU
		cores = to_uint((float)compute_units*(nvidia+amd+intel+arm)); // for CPUs, compute_units is the number of threads (twice the number of cores with hyperthreading)
		tflops = 1E-6f*(float)cores*(float)ipc*(float)clock_frequency; // estimated device floating point performance in TeraFLOPs/s
	}
};

string get_opencl_c_code(); // implemented in kernel.hpp
inline void print_device_info(const Device_Info& d, const int id=-1) { // print OpenCL device info
	println("\r|----------------.--------------------------------------------------|");
	if(id>-1) println("| Device ID      | "+alignl(48, to_string(id))+" |");
	println("| Device Name    | "+alignl(48, d.name                 )+" |");
	println("| Device Vendor  | "+alignl(48, d.vendor               )+" |");
	println("| Device Driver  | "+alignl(48, d.driver_version       )+" |");
	println("| OpenCL Version | "+alignl(48, d.opencl_c_version     )+" |");
	println("| Compute Units  | "+alignl(48, to_string(d.compute_units)+" at "+to_string(d.clock_frequency)+" MHz ("+to_string(d.cores)+" cores, "+to_string(d.tflops, 3)+" TFLOPs/s)")+" |");
	println("| Memory, Cache  | "+alignl(48, to_string(d.memory)+" MB, "+to_string(d.global_cache)+" KB global / "+to_string(d.local_cache)+" KB local")+" |");
	println("| Max Alloc Size | "+alignl(48, to_string(d.max_global_buffer)+" MB global, "+to_string(d.max_constant_buffer)+" KB constant")+" |");
	println("|----------------'--------------------------------------------------|");
}
inline vector<Device_Info> get_devices() { // returns a vector of all available OpenCL devices
	vector<Device_Info> devices; // get all devices of all platforms
	vector<cl::Platform> cl_platforms; // get all platforms (drivers)
	cl::Platform::get(&cl_platforms);
	for(uint i=0u; i<(uint)cl_platforms.size(); i++) {
		vector<cl::Device> cl_devices;
		cl_platforms[i].getDevices(CL_DEVICE_TYPE_ALL, &cl_devices);
		for(uint j=0u; j<(uint)cl_devices.size(); j++) {
			devices.push_back(Device_Info(cl_devices[j]));
		}
	}
	if((uint)cl_platforms.size()==0u||(uint)devices.size()==0u) {
		print_error("There are no OpenCL devices available. Make sure that the OpenCL 1.2 Runtime for your device is installed. For GPUs it comes by default with the graphics driver, for CPUs it has to be installed separately.");
	}
	println("\r|----------------.--------------------------------------------------|");
	for(uint i=0u; i<(uint)devices.size(); i++) {
		println("| Device ID "+alignr(4u, i)+" | "+alignl(48u, devices[i].name)+" |");
	}
	println("|----------------'--------------------------------------------------|");
	return devices;
}
inline Device_Info select_device_with_most_flops(const vector<Device_Info>& devices=get_devices()) { // returns device with best floating-point performance
	float best_value = 0.0f;
	uint best_i = 0u; // index of fastest device
	for(uint i=0u; i<(uint)devices.size(); i++) { // find device with highest (estimated) floating point performance
		if(devices[i].tflops>best_value) { // device_memory>best_value
			best_value = devices[i].tflops; // best_value = device_memory;
			best_i = i; // find index of fastest device
		}
	}
	print_device_info(devices[best_i], best_i); // print device info
	return devices[best_i];
}
inline Device_Info select_device_with_most_memory(const vector<Device_Info>& devices=get_devices()) { // returns device with largest memory capacity
	uint best_value = 0u;
	uint best_i = 0u; // index of fastest device
	for(uint i=0u; i<(uint)devices.size(); i++) { // find device with most memory
		if(devices[i].memory>best_value) {
			best_value = devices[i].memory;
			best_i = i; // find index of fastest device
		}
	}
	print_device_info(devices[best_i], best_i); // print device info
	return devices[best_i];
}
inline Device_Info select_device_with_id(const uint id, const vector<Device_Info>& devices=get_devices()) { // returns device with specified ID
	if(id<(uint)devices.size()) {
		print_device_info(devices[id], id); // print device info
		return devices[id];
	} else {
		print_error("Your selected device ID ("+to_string(id)+") is wrong. Check the setting \"#define DEVICE x\" in defines.hpp.");
		return devices[0]; // is never executed, just to avoid compiler warnings
	}
}

class Device {
private:
	cl::Context cl_context;
	cl::Program cl_program;
	cl::CommandQueue cl_queue;
	inline string enable_device_capabilities() const { return // enable FP64/FP16 capabilities if available
		"\n	#ifdef cl_khr_fp64"
		"\n	#pragma OPENCL EXTENSION cl_khr_fp64 : enable" // make sure cl_khr_fp64 extension is enabled
		"\n	#endif"
		"\n	#ifdef cl_khr_fp16"
		"\n	#pragma OPENCL EXTENSION cl_khr_fp16 : enable" // make sure cl_khr_fp16 extension is enabled
		"\n	#endif"
		"\n	#ifdef cl_khr_int64_base_atomics"
		"\n	#pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable" // make sure cl_khr_int64_base_atomics extension is enabled
		"\n	#endif"
	;}
public:
	Device_Info info;
	inline Device(const Device_Info& d=select_device_with_most_flops(), const string& opencl_c_code=get_opencl_c_code()) {
		info = d;
		cl_context = cl::Context(info.cl_device);
		cl_queue = cl::CommandQueue(cl_context, info.cl_device); // queue to push commands for the device
		cl::Program::Sources cl_source;
		const string kernel_code = enable_device_capabilities()+"\n"+opencl_c_code;
		cl_source.push_back({ kernel_code.c_str(), kernel_code.length() });
		cl_program = cl::Program(cl_context, cl_source);
#ifndef LOG
		int error = cl_program.build("-cl-fast-relaxed-math -w"); // compile OpenCL C code, disable warnings
		if(error) print_warning(cl_program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(info.cl_device)); // print build log
#else // LOG, generate logfile for OpenCL code compilation
		int error = cl_program.build("-cl-fast-relaxed-math"); // compile OpenCL C code
		const string log = cl_program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(info.cl_device);
		write_file("bin/kernel.log", log); // save build log
		if((uint)log.length()>2u) print_warning(log); // print build log
#endif // LOG
		if(error) print_error("OpenCL C code compilation failed. Make sure there are no errors in kernel.cpp (\"#define LOG\" might help). If your GPU is old, try uncommenting \"#define USE_OPENCL_1_1\".");
		else print_info("OpenCL C code successfully compiled.");
#ifdef PTX // generate assembly (ptx) file for OpenCL code
		write_file("bin/kernel.ptx", cl_program.getInfo<CL_PROGRAM_BINARIES>()[0]); // save binary (ptx file)
#endif // PTX
	}
	inline cl::Context get_cl_context() const {
		return cl_context;
	}
	inline cl::Program get_cl_program() const {
		return cl_program;
	}
	inline cl::CommandQueue get_cl_queue() const {
		return cl_queue;
	}
};

template<typename T> class Memory {
private:
	ulong N = 0ull; // buffer length
	uint d = 1u; // buffer dimensions
	bool host_buffer_exists = false;
	bool device_buffer_exists = false;
	T* host_buffer = nullptr; // host buffer
	cl::Buffer device_buffer; // device buffer
	cl::CommandQueue cl_queue;
	Device_Info* info; // pointer to Device_Info of linked Device (for tracking memory usage)
	void initialize_auxiliary_pointers() {
		x = s0 = host_buffer;
		if(d>0x1u) y = s1 = host_buffer+N;
		if(d>0x2u) z = s2 = host_buffer+N*0x2ull;
		if(d>0x3u) w = s3 = host_buffer+N*0x3ull;
		if(d>0x4u) s4 = host_buffer+N*0x4ull;
		if(d>0x5u) s5 = host_buffer+N*0x5ull;
		if(d>0x6u) s6 = host_buffer+N*0x6ull;
		if(d>0x7u) s7 = host_buffer+N*0x7ull;
		if(d>0x8u) s8 = host_buffer+N*0x8ull;
		if(d>0x9u) s9 = host_buffer+N*0x9ull;
		if(d>0xAu) sA = host_buffer+N*0xAull;
		if(d>0xBu) sB = host_buffer+N*0xBull;
		if(d>0xCu) sC = host_buffer+N*0xCull;
		if(d>0xDu) sD = host_buffer+N*0xDull;
		if(d>0xEu) sE = host_buffer+N*0xEull;
		if(d>0xFu) sF = host_buffer+N*0xFull;
	}
	void allocate_device_buffer(Device& device, const bool allocate_device) {
		if(N*(ulong)d==0ull) print_error("Memory size must be larger than 0.");
		info = &(device.info);
		cl_queue = device.get_cl_queue();
		if(allocate_device) {
			info->memory_used += (uint)(capacity()/1048576ull); // track device memory usage
			if(info->memory_used>info->memory) print_error("Device \""+info->name+"\" does not have enough memory. Allocating another "+to_string((uint)(capacity()/1048576ull))+" MB would use a total of "+to_string(info->memory_used)+" MB / "+to_string(info->memory)+" MB.");
			int error = 0;
			device_buffer = cl::Buffer(device.get_cl_context(), CL_MEM_READ_WRITE, capacity(), nullptr, &error);
			if(error) print_error("OpenCL Buffer allocation failed with error code "+to_string(error)+".");
			device_buffer_exists = true;
		}
	}
public:
	T *x=nullptr, *y=nullptr, *z=nullptr, *w=nullptr; // host buffer auxiliary pointers for multi-dimensional array access (array of structurs)
	T *s0=nullptr, *s1=nullptr, *s2=nullptr, *s3=nullptr, *s4=nullptr, *s5=nullptr, *s6=nullptr, *s7=nullptr, *s8=nullptr, *s9=nullptr, *sA=nullptr, *sB=nullptr, *sC=nullptr, *sD=nullptr, *sE=nullptr, *sF=nullptr;
	inline Memory(Device& device, const ulong N, const uint dimensions=1u, const bool allocate_host=true, const bool allocate_device=true) {
		this->N = N;
		this->d = dimensions;
		allocate_device_buffer(device, allocate_device);
		if(allocate_host) {
			host_buffer = new T[N*(ulong)dimensions];
			for(ulong i=0ull; i<N*(ulong)dimensions; i++) host_buffer[i] = (T)0;
			initialize_auxiliary_pointers();
			host_buffer_exists = true;
		}
		write_to_device();
	}
	inline Memory(Device& device, const ulong N, const uint dimensions, T* const host_buffer, const bool allocate_device=true) {
		this->N = N;
		this->d = dimensions;
		allocate_device_buffer(device, allocate_device);
		this->host_buffer = host_buffer;
		initialize_auxiliary_pointers();
		host_buffer_exists = true;
		write_to_device();
	}
	inline ~Memory() {
		delete[] host_buffer;
		device_buffer = nullptr;
		if(device_buffer_exists) info->memory_used -= (uint)(capacity()/1048576ull); // track device memory usage
	}
	inline void reset(const T value=(T)0) {
		if(host_buffer_exists) for(ulong i=0ull; i<N*(ulong)d; i++) host_buffer[i] = value;
		write_to_device();
	}
	inline const ulong length() const {
		return N;
	}
	inline const uint dimensions() const {
		return d;
	}
	inline const ulong range() const {
		return N*(ulong)d;
	}
	inline const ulong capacity() const { // returns capacity of the buffer in Byte
		return N*(ulong)d*sizeof(T);
	}
	inline T* const data() {
		return host_buffer;
	}
	inline const T* const data() const {
		return host_buffer;
	}
	inline T* const operator()() {
		return host_buffer;
	}
	inline const T* const operator()() const {
		return host_buffer;
	}
	inline T& operator[](const ulong i) {
		return host_buffer[i];
	}
	inline const T& operator[](const ulong i) const {
		return host_buffer[i];
	}
	inline const T operator()(const ulong i) const {
		return host_buffer[i];
	}
	inline const T operator()(const ulong i, const uint dimension) const {
		return host_buffer[i+(ulong)dimension*N]; // array of structurs
	}
	inline void read_from_device(const bool blocking=true) {
		if(host_buffer_exists&&device_buffer_exists) cl_queue.enqueueReadBuffer(device_buffer, blocking, 0u, capacity(), (void*)host_buffer);
	}
	inline void write_to_device(const bool blocking=true) {
		if(host_buffer_exists&&device_buffer_exists) cl_queue.enqueueWriteBuffer(device_buffer, blocking, 0u, capacity(), (void*)host_buffer);
	}
	inline void read_from_device(const ulong offset, const ulong length, const bool blocking=true) {
		if(host_buffer_exists&&device_buffer_exists) cl_queue.enqueueReadBuffer(device_buffer, blocking, min(offset, N*(ulong)d)*sizeof(T), min(length, N*(ulong)d-offset)*sizeof(T), (void*)host_buffer);
	}
	inline void write_to_device(const ulong offset, const ulong length, const bool blocking=true) {
		if(host_buffer_exists&&device_buffer_exists) cl_queue.enqueueWriteBuffer(device_buffer, blocking, min(offset, N*(ulong)d)*sizeof(T), min(length, N*(ulong)d-offset)*sizeof(T), (void*)host_buffer);
	}
	inline void finish() {
		cl_queue.finish();
	}
	inline const cl::Buffer& get_cl_buffer() const {
		return device_buffer;
	}
};

class Kernel {
private:
	uint number_of_parameters = 0u;
	cl::Kernel cl_kernel;
	cl::NDRange cl_range_global, cl_range_local;
	cl::CommandQueue cl_queue;
	inline void initialize_ranges(const ulong N) {
		cl_range_local = cl::NDRange(THREAD_BLOCK_SIZE); // warp size is 32
		cl_range_global = cl::NDRange(((N+THREAD_BLOCK_SIZE-1)/THREAD_BLOCK_SIZE)*THREAD_BLOCK_SIZE); // make global range a multiple of local range
	}
public:
	template<typename... T> inline Kernel(const Device& device, const ulong N, const string& name, const Memory<T>&... parameters) {
		cl_kernel = cl::Kernel(device.get_cl_program(), name.c_str());
		(cl_kernel.setArg(number_of_parameters++, parameters.get_cl_buffer()), ...); // expand variadic template to link buffers against kernel parameters
		initialize_ranges(N);
		cl_queue = device.get_cl_queue();
	}
	template<typename... T> inline Kernel(const Device& device, const ulong N, const string& name, const Memory<T>*... parameters) {
		cl_kernel = cl::Kernel(device.get_cl_program(), name.c_str());
		(cl_kernel.setArg(number_of_parameters++, parameters->get_cl_buffer()), ...); // expand variadic template to link buffers against kernel parameters
		initialize_ranges(N);
		cl_queue = device.get_cl_queue();
	}
	template<typename... T> inline void add_parameters(const Memory<T>&... parameters) {
		(cl_kernel.setArg(number_of_parameters++, parameters.get_cl_buffer()), ...); // expand variadic template to link buffers against kernel parameters
	}
	template<typename... T> inline void add_parameters(const Memory<T>*... parameters) {
		(cl_kernel.setArg(number_of_parameters++, parameters->get_cl_buffer()), ...); // expand variadic template to link buffers against kernel parameters
	}
	template<typename... T> inline void add_constants(const T... constants) {
		(cl_kernel.setArg(number_of_parameters++, constants), ...); // expand variadic template to pass constants as kernel parameters
	}
	inline void run() const {
		cl_queue.enqueueNDRangeKernel(cl_kernel, cl::NullRange, cl_range_global, cl_range_local);
		cl_queue.finish();
	}
};