#define _CRT_SECURE_NO_WARNINGS
#define MAX_SOURCE_SIZE (0x100000)
#define GPU_CPU 0 // 0-gpu, 1-cpu
#include <CL/cl.h>
#include <iostream>

std::pair<char*, size_t> read_kernel() {
	FILE* fp;
	char* source_str;
	size_t source_size;
	fp = fopen("kernel.cl", "r");
	if (!fp) {
		fprintf(stderr, "Failed to load kernel.\n");
		exit(1);
	}
	source_str = (char*)malloc(MAX_SOURCE_SIZE);
	source_size = fread(source_str, 1, MAX_SOURCE_SIZE, fp);
	fclose(fp);
	return std::pair<char*, size_t>{source_str, source_size};
}
void print_device(cl_context context) {
	size_t size = 0;
	clGetContextInfo(context, CL_CONTEXT_DEVICES, 0, NULL, &size);
	cl_device_id device = NULL;
	if (size > 0) {
		cl_device_id* devices = (cl_device_id*)alloca(size);
		clGetContextInfo(context, CL_CONTEXT_DEVICES, size, devices, NULL);
		device = devices[0];
		char deviceName[100];
		size_t real_size{};
		clGetDeviceInfo(device, CL_DEVICE_NAME, 100, deviceName, &real_size);
		std::cout << "" << deviceName << std::endl;
	}
}
void print_info() {
	cl_uint platformCount = 0;
	clGetPlatformIDs(0, nullptr, &platformCount);
	cl_platform_id* platform = new cl_platform_id[platformCount];
	clGetPlatformIDs(platformCount, platform, nullptr);
	for (cl_uint i = 0; i < platformCount; ++i) {
		char platformName[128];
		clGetPlatformInfo(platform[i], CL_PLATFORM_NAME,
			128, platformName, nullptr);
		std::cout << platformName << std::endl;
	}
	std::cout << "\n\n";
}

int main() {
	print_info();

	const int SIZE = 11;
	int* data = new int[SIZE];
	int* res = new int[SIZE];
	for (int i = 0; i < SIZE; i++) {
		data[i] = i;
	}

	cl_int err;
	auto src_pair = read_kernel();
	auto source_str = src_pair.first;
	auto source_size = src_pair.second;

	cl_device_id device_id = NULL;
	cl_uint num_platforms;
	cl_uint num_devices;
	clGetPlatformIDs(0, NULL, &num_platforms);
	cl_platform_id* platforms = new cl_platform_id[num_platforms];
	clGetPlatformIDs(num_platforms, platforms, NULL);
	// [0] - gpu, [1] - cpu
	clGetDeviceIDs(platforms[GPU_CPU], CL_DEVICE_TYPE_ALL, 1, &device_id, &num_devices);
	cl_context context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &err);
	// std::cout << "clCreateContext err: " << err << std::endl;
	print_device(context);
	cl_command_queue queue = clCreateCommandQueue(context, device_id, 0, &err);
	// std::cout << "clCreateCommandQueue err: " << err << std::endl;

	cl_mem input = clCreateBuffer(context, CL_MEM_READ_ONLY, SIZE * sizeof(int), NULL, NULL);
	cl_mem output = clCreateBuffer(context, CL_MEM_WRITE_ONLY, SIZE * sizeof(int), NULL, NULL);
	clEnqueueWriteBuffer(queue, input, CL_TRUE, 0, SIZE * sizeof(int), data, 0, NULL, NULL);

	cl_program program = clCreateProgramWithSource(context, 1, (const char**)&source_str, (const size_t*)&source_size, &err);
	// std::cout << "clCreateProgramWithSource err: " << err << std::endl;
	err = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);
	// std::cout << "clBuildProgram err: " << err << std::endl;
	cl_program_build_info build_info;
	char* log = new char[10000];
	clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, 10000 * sizeof(char), log, NULL);
	std::cout << log << std::endl;

	cl_kernel kernel = clCreateKernel(program, "sum", &err);
	// std::cout << "clCreateKernel err: " << err << std::endl;
	clSetKernelArg(kernel, 0, sizeof(cl_mem), &input);
	clSetKernelArg(kernel, 1, sizeof(cl_mem), &output);
	clSetKernelArg(kernel, 2, sizeof(int), &SIZE);

	size_t global_item_size = SIZE;
	size_t local_item_size = 0;
	// must be: global % local = 0
	clGetKernelWorkGroupInfo(kernel, device_id, CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), &local_item_size, NULL);
	// std::cout << local_item_size << std::endl;
	err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global_item_size,
		(local_item_size <= global_item_size && global_item_size % local_item_size == 0) ? &local_item_size : NULL, 0, NULL, NULL);
	// std::cout << "clEnqueueNDRangeKernel err: " << err << std::endl;
	clFinish(queue);
	clEnqueueReadBuffer(queue, output, CL_TRUE, 0, sizeof(float) * SIZE, res, 0, NULL, NULL);
	for (int i = 0; i < SIZE; i++) {
		std::cout << data[i] << " + global_id = " << res[i] << std::endl;
	}
	clReleaseMemObject(input);
	clReleaseMemObject(output);
	clReleaseProgram(program);
	clReleaseKernel(kernel);
	clReleaseCommandQueue(queue);
	clReleaseContext(context);
	return 0;
}