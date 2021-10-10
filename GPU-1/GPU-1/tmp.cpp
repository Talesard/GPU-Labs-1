//#define _CRT_SECURE_NO_WARNINGS
//#include <CL/cl.h>
//#include <iostream>
//#define MAX_SOURCE_SIZE (0x100000)
//
//std::pair<char*, size_t> read_kernel() {
//	FILE* fp;
//	char* source_str;
//	size_t source_size;
//	fp = fopen("kernel.cl", "r");
//	if (!fp) {
//		fprintf(stderr, "Failed to load kernel.\n");
//		exit(1);
//	}
//	source_str = (char*)malloc(MAX_SOURCE_SIZE);
//	source_size = fread(source_str, 1, MAX_SOURCE_SIZE, fp);
//	fclose(fp);
//	return std::pair<char*, size_t>{source_str, source_size};
//}
//
//int main() {
//	cl_int err;
//	auto src_pair = read_kernel();
//	auto source_str = src_pair.first;
//	auto source_size = src_pair.second;
//
//	cl_uint numPlatforms = 0;
//	clGetPlatformIDs(0, NULL, &numPlatforms);
//	cl_platform_id platform = NULL;
//	if (0 < numPlatforms) {
//		cl_platform_id* platforms = new cl_platform_id[numPlatforms];
//		clGetPlatformIDs(numPlatforms, platforms, NULL);
//		platform = platforms[0];
//	}
//	cl_context_properties properties[3] = { CL_CONTEXT_PLATFORM, (cl_context_properties)platform, 0 };
//	cl_context context = clCreateContextFromType(NULL == platform ? NULL : properties, CL_DEVICE_TYPE_GPU, NULL, NULL, &err);
//	std::cout << "clCreateContextFromType err: " << err << std::endl;
//	size_t size = 0;
//	clGetContextInfo(context, CL_CONTEXT_DEVICES, 0, NULL, &size);
//	cl_device_id device = NULL;
//	if (size > 0) {
//		cl_device_id* devices = (cl_device_id*)alloca(size);
//		clGetContextInfo(context, CL_CONTEXT_DEVICES, size, devices, NULL);
//		device = devices[0];
//		char deviceName[100];
//		size_t real_size{};
//		clGetDeviceInfo(device, CL_DEVICE_NAME, 100, deviceName, &real_size);
//		std::cout << "Using: " << deviceName << std::endl;
//	}
//
//	cl_command_queue queue = clCreateCommandQueue(context, device, 0, &err);
//	std::cout << "clCreateCommandQueue err: " << err << std::endl;
//	cl_program program = clCreateProgramWithSource(context, 1, (const char**)&source_str, (const size_t*)&source_size, &err);
//	std::cout << "clCreateProgramWithSource err: " << err << std::endl;
//	clBuildProgram(program, 1, &device, NULL, NULL, NULL);
//	cl_program_build_info build_info;
//	char* log = new char[10000];
//	clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 10000 * sizeof(char), log, NULL);
//	std::cout << log << std::endl;
//	cl_kernel kernel = clCreateKernel(program, "square", &err);
//	std::cout << "clCreateKernel err: " << err << std::endl;
//	const int SIZE = 10;
//	float data[SIZE];
//	float res[SIZE];
//	for (int i = 0; i < SIZE; i++) {
//		data[i] = rand();
//	}
//	cl_mem input = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * SIZE, NULL, &err);
//	std::cout << "clCreateBuffer_input err: " << err << std::endl;
//	cl_mem output = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * SIZE, NULL, &err);
//	std::cout << "clCreateBuffer_output err: " << err << std::endl;
//	clEnqueueWriteBuffer(queue, input, CL_TRUE, 0, sizeof(float) * SIZE, data, 0, NULL, NULL);
//	const size_t count = SIZE;
//	clSetKernelArg(kernel, 0, sizeof(cl_mem), &input);
//	clSetKernelArg(kernel, 1, sizeof(cl_mem), &output);
//	clSetKernelArg(kernel, 2, sizeof(size_t), &count);
//	size_t group;
//	clGetKernelWorkGroupInfo(kernel, device, CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), &group, NULL);
//	clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &count, &group, 0, NULL, NULL);
//	clFinish(queue);
//
//	clReleaseMemObject(input);
//	clReleaseMemObject(output);
//	clReleaseProgram(program);
//	clReleaseKernel(kernel);
//	clReleaseCommandQueue(queue);
//	clReleaseContext(context);
//	return 0;
//}