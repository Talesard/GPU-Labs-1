#pragma once
#include <iostream>
#include <omp.h>
#include <CL/cl.h>
#include "ocl_helpers.h"
// #define DEBUG

void mult_classic_cpu(float* A, float* B, float* Res,  int m, int n, int k) {
	for (int i = 0; i < m; i++) {
		for (int j = 0; j < k; j++) {
			float sum = 0;
			for (int r = 0; r < n; r++) {
				sum += A[i * n + r] * B[j + k * r];
			}
			Res[k * i + j] = sum;
		}
	}
}

void mult_classic_cpu_omp(float* A, float* B, float* Res, int m, int n, int k) {
#pragma omp parallel for
	for (int i = 0; i < m; i++) {
		for (int j = 0; j < k; j++) {
			float sum = 0;
			for (int r = 0; r < n; r++) {
				sum += A[i * n + r] * B[j + k * r];
			}
			Res[k * i + j] = sum;
		}
	}
}

void mult_classic_cl(float* A, float* B, float* Res, int m, int n, int k, int gpu_cpu) {
	cl_int err;
	auto src_pair = read_kernel("kernel.cl");
	auto source_str = src_pair.first;
	auto source_size = src_pair.second;

	cl_device_id device_id = NULL;
	cl_uint num_platforms;
	cl_uint num_devices;
	clGetPlatformIDs(0, NULL, &num_platforms);
	cl_platform_id* platforms = new cl_platform_id[num_platforms];
	clGetPlatformIDs(num_platforms, platforms, NULL);
	clGetDeviceIDs(platforms[gpu_cpu], CL_DEVICE_TYPE_ALL, 1, &device_id, &num_devices);
	cl_context context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &err);
	print_device(context);
	cl_command_queue queue = clCreateCommandQueue(context, device_id, 0, &err);

	cl_program program = clCreateProgramWithSource(context, 1, (const char**)&source_str, (const size_t*)&source_size, &err);
	err = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);

#ifdef DEBUG
	std::cout << "clBuildProgram err: " << err << std::endl;
	cl_program_build_info build_info;
	char* log = new char[10000];
	clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, 10000 * sizeof(char), log, NULL);
	std::cout << log << std::endl;
#endif // DEBUG

	cl_kernel kernel = clCreateKernel(program, "mult_classic", &err);

	cl_mem input_A = clCreateBuffer(context, CL_MEM_READ_ONLY, m * n * sizeof(float), NULL, NULL);
	cl_mem input_B = clCreateBuffer(context, CL_MEM_READ_ONLY, n * k * sizeof(float), NULL, NULL);
	cl_mem output_Res = clCreateBuffer(context, CL_MEM_READ_WRITE, m * k * sizeof(float), NULL, NULL);

	clEnqueueWriteBuffer(queue, input_A, CL_TRUE, 0, m * n * sizeof(float), A, 0, NULL, NULL);
	clEnqueueWriteBuffer(queue, input_B, CL_TRUE, 0, n * k * sizeof(float), B, 0, NULL, NULL);
	clEnqueueWriteBuffer(queue, output_Res, CL_TRUE, 0, m * k * sizeof(float), Res, 0, NULL, NULL);

	clSetKernelArg(kernel, 0, sizeof(cl_mem), &input_A);
	clSetKernelArg(kernel, 1, sizeof(cl_mem), &input_B);
	clSetKernelArg(kernel, 2, sizeof(cl_mem), &output_Res);
	clSetKernelArg(kernel, 3, sizeof(int), &m);
	clSetKernelArg(kernel, 4, sizeof(int), &n);
	clSetKernelArg(kernel, 5, sizeof(int), &k);

	size_t block_size[] = { 16, 16 };
	size_t global_size[] = { m, k };

	auto t0 = omp_get_wtime();
	clEnqueueNDRangeKernel(queue, kernel, 2, NULL, global_size, block_size, 0, NULL, NULL);
	clFinish(queue);
	auto t1 = omp_get_wtime();

	std::string msg = "";
	if (gpu_cpu == 0) msg = "Time classic GPU cl: ";
	if (gpu_cpu == 1) msg = "Time classic CPU cl: ";
	std::cout << msg << t1 - t0 << std::endl;

	clEnqueueReadBuffer(queue, output_Res, CL_TRUE, 0, m * k * sizeof(float), Res, 0, NULL, NULL);

	clReleaseMemObject(input_A);
	clReleaseMemObject(input_B);
	clReleaseMemObject(output_Res);
	clReleaseProgram(program);
	clReleaseKernel(kernel);
	clReleaseCommandQueue(queue);
	clReleaseContext(context);
}

void mult_block_cl(float* A, float* B, float* Res, int m, int n, int k, int gpu_cpu) {
	cl_int err;
	auto src_pair = read_kernel("kernel.cl");
	auto source_str = src_pair.first;
	auto source_size = src_pair.second;

	cl_device_id device_id = NULL;
	cl_uint num_platforms;
	cl_uint num_devices;
	clGetPlatformIDs(0, NULL, &num_platforms);
	cl_platform_id* platforms = new cl_platform_id[num_platforms];
	clGetPlatformIDs(num_platforms, platforms, NULL);
	clGetDeviceIDs(platforms[gpu_cpu], CL_DEVICE_TYPE_ALL, 1, &device_id, &num_devices);
	cl_context context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &err);
	print_device(context);
	cl_command_queue queue = clCreateCommandQueue(context, device_id, 0, &err);

	cl_program program = clCreateProgramWithSource(context, 1, (const char**)&source_str, (const size_t*)&source_size, &err);
	err = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);

#ifdef DEBUG
	std::cout << "clBuildProgram err: " << err << std::endl;
	cl_program_build_info build_info;
	char* log = new char[10000];
	clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, 10000 * sizeof(char), log, NULL);
	std::cout << log << std::endl;
#endif // DEBUG

	cl_kernel kernel = clCreateKernel(program, "mult_block", &err);

	cl_mem input_A = clCreateBuffer(context, CL_MEM_READ_ONLY, m * n * sizeof(float), NULL, NULL);
	cl_mem input_B = clCreateBuffer(context, CL_MEM_READ_ONLY, n * k * sizeof(float), NULL, NULL);
	cl_mem output_Res = clCreateBuffer(context, CL_MEM_READ_WRITE, m * k * sizeof(float), NULL, NULL);

	clEnqueueWriteBuffer(queue, input_A, CL_TRUE, 0, m * n * sizeof(float), A, 0, NULL, NULL);
	clEnqueueWriteBuffer(queue, input_B, CL_TRUE, 0, n * k * sizeof(float), B, 0, NULL, NULL);
	clEnqueueWriteBuffer(queue, output_Res, CL_TRUE, 0, m * k * sizeof(float), Res, 0, NULL, NULL);

	clSetKernelArg(kernel, 0, sizeof(cl_mem), &input_A);
	clSetKernelArg(kernel, 1, sizeof(cl_mem), &input_B);
	clSetKernelArg(kernel, 2, sizeof(cl_mem), &output_Res);
	clSetKernelArg(kernel, 3, sizeof(int), &m);
	clSetKernelArg(kernel, 4, sizeof(int), &n);
	clSetKernelArg(kernel, 5, sizeof(int), &k);

	size_t block_size[] = { 16, 16 };
	size_t global_size[] = { m, k };

	auto t0 = omp_get_wtime();
	clEnqueueNDRangeKernel(queue, kernel, 2, NULL, global_size, block_size, 0, NULL, NULL);
	clFinish(queue);
	auto t1 = omp_get_wtime();

	std::string msg = "";
	if (gpu_cpu == 0) msg = "Time block GPU cl: ";
	if (gpu_cpu == 1) msg = "Time block CPU cl: ";
	std::cout << msg << t1 - t0 << std::endl;

	clEnqueueReadBuffer(queue, output_Res, CL_TRUE, 0, m * k * sizeof(float), Res, 0, NULL, NULL);

	clReleaseMemObject(input_A);
	clReleaseMemObject(input_B);
	clReleaseMemObject(output_Res);
	clReleaseProgram(program);
	clReleaseKernel(kernel);
	clReleaseCommandQueue(queue);
	clReleaseContext(context);
}

void mult_image_cl(float* A, float* B, float* Res, int m, int n, int k, int gpu_cpu) {
	cl_int err;
	auto src_pair = read_kernel("kernel.cl");
	auto source_str = src_pair.first;
	auto source_size = src_pair.second;

	cl_device_id device_id = NULL;
	cl_uint num_platforms;
	cl_uint num_devices;
	clGetPlatformIDs(0, NULL, &num_platforms);
	cl_platform_id* platforms = new cl_platform_id[num_platforms];
	clGetPlatformIDs(num_platforms, platforms, NULL);
	clGetDeviceIDs(platforms[gpu_cpu], CL_DEVICE_TYPE_ALL, 1, &device_id, &num_devices);
	cl_context context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &err);
	print_device(context);
	cl_command_queue queue = clCreateCommandQueue(context, device_id, 0, &err);

	cl_program program = clCreateProgramWithSource(context, 1, (const char**)&source_str, (const size_t*)&source_size, &err);
	err = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);

#ifdef DEBUG
	std::cout << "clBuildProgram err: " << err << std::endl;
	cl_program_build_info build_info;
	char* log = new char[10000];
	clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, 10000 * sizeof(char), log, NULL);
	std::cout << log << std::endl;
#endif // DEBUG

	cl_kernel kernel = clCreateKernel(program, "mult_image", &err);

	cl_image_format format;
	format.image_channel_order = CL_R; // одноканальная картинка
	format.image_channel_data_type = CL_FLOAT; // тип картинки
	
	// Origin: Defines the(x, y, z) offset in pixels in the image from where to write or write.
	// If image is a 2D image object, the z value given by origin[2] must be 0.
	size_t origin[] = { 0, 0, 0 };

	cl_image_desc descA = {}; // описание типа изображения
	descA.image_type = CL_MEM_OBJECT_IMAGE2D;
	descA.image_width = m;
	descA.image_height = n;

	cl_image_desc descB = {};
	descB.image_type = CL_MEM_OBJECT_IMAGE2D;
	descB.image_width = n;
	descB.image_height = k;

	cl_image_desc descRes = {};
	descRes.image_type = CL_MEM_OBJECT_IMAGE2D;
	descRes.image_width = m;
	descRes.image_height = k;

	cl_mem input_A = clCreateImage(context, CL_MEM_READ_ONLY, &format, &descA, NULL, NULL);
	cl_mem input_B = clCreateImage(context, CL_MEM_READ_ONLY, &format, &descB, NULL, NULL);
	cl_mem output_Res = clCreateImage(context, CL_MEM_WRITE_ONLY, &format, &descRes, NULL, NULL);

	// Region: Defines the (width, height, depth) in pixels of the 2D or 3D rectangle being write or written.
	// If image is a 2D image object, the depth value given by region[2] must be 1. 
	size_t regionA[] = { m, n, 1 };
	clEnqueueWriteImage(queue, input_A, CL_TRUE, origin, regionA, 0, 0, A, 0, NULL, NULL);
	size_t regionB[] = { n, k, 1 };
	clEnqueueWriteImage(queue, input_B, CL_TRUE, origin, regionB, 0, 0, B, 0, NULL, NULL);


	clSetKernelArg(kernel, 0, sizeof(cl_mem), &input_A);
	clSetKernelArg(kernel, 1, sizeof(cl_mem), &input_B);
	clSetKernelArg(kernel, 2, sizeof(cl_mem), &output_Res);
	clSetKernelArg(kernel, 3, sizeof(int), &m);
	clSetKernelArg(kernel, 4, sizeof(int), &n);
	clSetKernelArg(kernel, 5, sizeof(int), &k);

	size_t block_size[] = { 16, 16 };
	size_t global_size[] = { m, k };

	auto t0 = omp_get_wtime();
	clEnqueueNDRangeKernel(queue, kernel, 2, NULL, global_size, block_size, 0, NULL, NULL);
	clFinish(queue);
	auto t1 = omp_get_wtime();

	std::string msg = "";
	if (gpu_cpu == 0) msg = "Time image GPU cl: ";
	if (gpu_cpu == 1) msg = "Time image CPU cl: ";
	std::cout << msg << t1 - t0 << std::endl;

	size_t regionRes[] = { m, k, 1 };
	clEnqueueReadImage(queue, output_Res, CL_TRUE, origin, regionRes, 0, 0, Res, 0, NULL, NULL);

	clReleaseMemObject(input_A);
	clReleaseMemObject(input_B);
	clReleaseMemObject(output_Res);
	clReleaseProgram(program);
	clReleaseKernel(kernel);
	clReleaseCommandQueue(queue);
	clReleaseContext(context);
}
