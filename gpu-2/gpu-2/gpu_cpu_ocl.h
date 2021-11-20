#pragma once
#include "ocl_helpers.h"

void daxpy_gpu(int n, double a, double* x, int incx, double* y, int incy, int gpu_cpu, size_t local_sz) {
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
	clGetDeviceIDs(platforms[gpu_cpu], CL_DEVICE_TYPE_ALL, 1, &device_id, &num_devices);
	cl_context context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &err);
	print_device(context);
	//std::cout << "clCreateContext err: " << err << std::endl;
	// print_device(context);
	cl_command_queue queue = clCreateCommandQueue(context, device_id, 0, &err);
	//std::cout << "clCreateCommandQueue err: " << err << std::endl;

	cl_mem input_x = clCreateBuffer(context, CL_MEM_READ_ONLY, n * sizeof(double), NULL, NULL);
	cl_mem input_output_y = clCreateBuffer(context, CL_MEM_READ_WRITE, n * sizeof(double), NULL, NULL);
	err = clEnqueueWriteBuffer(queue, input_x, CL_TRUE, 0, n * sizeof(double), x, 0, NULL, NULL);
	// std::cout << "clEnqueueWriteBuffer input_x err: " << err << std::endl;
	err = clEnqueueWriteBuffer(queue, input_output_y, CL_TRUE, 0, n * sizeof(double), y, 0, NULL, NULL);
	// std::cout << "clEnqueueWriteBuffer input_output_y err: " << err << std::endl;
	if (err != 0) exit(-1);

	cl_program program = clCreateProgramWithSource(context, 1, (const char**)&source_str, (const size_t*)&source_size, &err);
	//std::cout << "clCreateProgramWithSource err: " << err << std::endl;
	err = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);
	//std::cout << "clBuildProgram err: " << err << std::endl;
	//cl_program_build_info build_info;
	//char* log = new char[10000];
	//clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, 10000 * sizeof(char), log, NULL);
	//std::cout << log << std::endl;


	cl_kernel kernel = clCreateKernel(program, "daxpy", &err);
	//std::cout << "clCreateKernel err: " << err << std::endl;
	err = clSetKernelArg(kernel, 0, sizeof(int), &n);
	//std::cout << "arg0 err: " << err << std::endl;
	err = clSetKernelArg(kernel, 1, sizeof(double), &a);
	//std::cout << "arg1 err: " << err << std::endl;
	err = clSetKernelArg(kernel, 2, sizeof(cl_mem), &input_x);
	//std::cout << "arg2 err: " << err << std::endl;
	err = clSetKernelArg(kernel, 3, sizeof(int), &incx);
	//std::cout << "arg3 err: " << err << std::endl;
	err = clSetKernelArg(kernel, 4, sizeof(cl_mem), &input_output_y);
	//std::cout << "arg4 err: " << err << std::endl;
	err = clSetKernelArg(kernel, 5, sizeof(int), &incy);
	//std::cout << "arg5 err: " << err << std::endl;


	size_t global_item_size = n;
	size_t local_item_size = local_sz;


	auto t0 = omp_get_wtime();
	err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global_item_size, n % local_item_size == 0 ? &local_item_size : NULL, 0, NULL, NULL);
	//std::cout << "clEnqueueNDRangeKernel err: " << err << std::endl;
	clFinish(queue);
	auto t1 = omp_get_wtime();
	std::string msg = "";
	if (gpu_cpu == 0) msg = "double GPU ocl only kernel: ";
	if (gpu_cpu == 1) msg = "double CPU ocl only kernel: ";
	std::cout << msg << t1 - t0 << std::endl;

	clEnqueueReadBuffer(queue, input_output_y, CL_TRUE, 0, sizeof(double) * n, y, 0, NULL, NULL);

#ifdef DEBUG
	for (int i = 0; i < n; i++) {
		std::cout << y[i] << " ";
	}
#endif // DEBUG

	clReleaseMemObject(input_x);
	clReleaseMemObject(input_output_y);
	clReleaseProgram(program);
	clReleaseKernel(kernel);
	clReleaseCommandQueue(queue);
	clReleaseContext(context);
}


void saxpy_gpu(int n, float a, float* x, int incx, float* y, int incy, int gpu_cpu, size_t local_sz) {
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
	clGetDeviceIDs(platforms[gpu_cpu], CL_DEVICE_TYPE_ALL, 1, &device_id, &num_devices);
	cl_context context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &err);
	print_device(context);
	//std::cout << "clCreateContext err: " << err << std::endl;
	// print_device(context);
	cl_command_queue queue = clCreateCommandQueue(context, device_id, 0, &err);
	//std::cout << "clCreateCommandQueue err: " << err << std::endl;

	cl_mem input_x = clCreateBuffer(context, CL_MEM_READ_ONLY, n * sizeof(float), NULL, NULL);
	cl_mem input_output_y = clCreateBuffer(context, CL_MEM_READ_WRITE, n * sizeof(float), NULL, NULL);
	err = clEnqueueWriteBuffer(queue, input_x, CL_TRUE, 0, n * sizeof(float), x, 0, NULL, NULL);
	// std::cout << "clEnqueueWriteBuffer input_x err: " << err << std::endl;
	err = clEnqueueWriteBuffer(queue, input_output_y, CL_TRUE, 0, n * sizeof(float), y, 0, NULL, NULL);
	// std::cout << "clEnqueueWriteBuffer input_output_y err: " << err << std::endl;

	cl_program program = clCreateProgramWithSource(context, 1, (const char**)&source_str, (const size_t*)&source_size, &err);
	//std::cout << "clCreateProgramWithSource err: " << err << std::endl;
	err = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);
	//std::cout << "clBuildProgram err: " << err << std::endl;
	//cl_program_build_info build_info;
	//char* log = new char[10000];
	//clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, 10000 * sizeof(char), log, NULL);
	//std::cout << log << std::endl;

	cl_kernel kernel = clCreateKernel(program, "saxpy", &err);
	//std::cout << "clCreateKernel err: " << err << std::endl;
	err = clSetKernelArg(kernel, 0, sizeof(int), &n);
	//std::cout << "arg0 err: " << err << std::endl;
	err = clSetKernelArg(kernel, 1, sizeof(float), &a);
	//std::cout << "arg1 err: " << err << std::endl;
	err = clSetKernelArg(kernel, 2, sizeof(cl_mem), &input_x);
	//std::cout << "arg2 err: " << err << std::endl;
	err = clSetKernelArg(kernel, 3, sizeof(int), &incx);
	//std::cout << "arg3 err: " << err << std::endl;
	err = clSetKernelArg(kernel, 4, sizeof(cl_mem), &input_output_y);
	//std::cout << "arg4 err: " << err << std::endl;
	err = clSetKernelArg(kernel, 5, sizeof(int), &incy);
	//std::cout << "arg5 err: " << err << std::endl;


	size_t global_item_size = n;

	size_t local_item_size = local_sz;
	// std::cout << (n % local_item_size == 0 ? local_item_size : 0) << std::endl;

	auto t0 = omp_get_wtime();
	err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global_item_size, n % local_item_size == 0 ? &local_item_size : NULL, 0, NULL, NULL);
	//std::cout << "clEnqueueNDRangeKernel err: " << err << std::endl;
	clFinish(queue);
	auto t1 = omp_get_wtime();
	std::string msg = "";
	if (gpu_cpu == 0) msg = "float GPU ocl only kernel: ";
	if (gpu_cpu == 1) msg = "float CPU ocl only kernel: ";
	std::cout << msg << t1 - t0 << std::endl;


	clEnqueueReadBuffer(queue, input_output_y, CL_TRUE, 0, sizeof(float) * n, y, 0, NULL, NULL);

#ifdef DEBUG
	for (int i = 0; i < n; i++) {
		std::cout << y[i] << " ";
	}
#endif // DEBUG

	clReleaseMemObject(input_x);
	clReleaseMemObject(input_output_y);
	clReleaseProgram(program);
	clReleaseKernel(kernel);
	clReleaseCommandQueue(queue);
	clReleaseContext(context);
}