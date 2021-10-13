#define _CRT_SECURE_NO_WARNINGS
#include <omp.h>
#include <iostream>
#include <vector>
#include <CL/cl.h>

#define MAX_SOURCE_SIZE (0x100000)
#define GPU_CPU 1
#define EPS 1e-3
#define GLOBAL_SIZE 256
// #define CHECKS
// #define DEBUG

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

void saxpy(int n, float a, float* x, int incx, float* y, int incy) {
	for (int i = 0; i < n; i++) {
		int id_y = i * incy;
		int id_x = i * incx;
		if ((id_y >= 0 && id_y < n) && (id_x >= 0 && id_x < n))
		y[id_y] = y[id_y] + a * x[id_x];
	}
}

void saxpy_omp(int n, float a, float* x, int incx, float* y, int incy) {
#pragma omp parallel for
	for (int i = 0; i < n; i++) {
		int id_y = i * incy;
		int id_x = i * incx;
		if ((id_y >= 0 && id_y < n) && (id_x >= 0 && id_x < n)) {
			y[id_y] = y[id_y] + a * x[id_x];
		}
	}
}

void daxpy(int n, double a, double* x, int incx, double* y, int incy) {
	for (int i = 0; i < n; i++) {
		int id_y = i * incy;
		int id_x = i * incx;
		if ((id_y >= 0 && id_y < n) && (id_x >= 0 && id_x < n))
		y[id_y] = y[id_y] + a * x[id_x];
	}
}

void daxpy_omp(int n, double a, double* x, int incx, double* y, int incy) {
#pragma omp parallel for
	for (int i = 0; i < n; i++) {
		int id_y = i * incy;
		int id_x = i * incx;
		if ((id_y >= 0 && id_y < n) && (id_x >= 0 && id_x < n)) {
			y[id_y] = y[id_y] + a * x[id_x];
		}
	}
}

void saxpy_gpu(int n, float a, float* x, int incx, float* y, int incy) {
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
	//std::cout << "clCreateContext err: " << err << std::endl;
	// print_device(context);
	cl_command_queue queue = clCreateCommandQueue(context, device_id, 0, &err);
	//std::cout << "clCreateCommandQueue err: " << err << std::endl;

	cl_mem input_x = clCreateBuffer(context, CL_MEM_READ_ONLY, n * sizeof(float), NULL, NULL);
	cl_mem input_output_y = clCreateBuffer(context, CL_MEM_READ_WRITE, n * sizeof(float), NULL, NULL);
	clEnqueueWriteBuffer(queue, input_x, CL_TRUE, 0, n * sizeof(float), x, 0, NULL, NULL);
	clEnqueueWriteBuffer(queue, input_output_y, CL_TRUE, 0, n * sizeof(float), y, 0, NULL, NULL);

	cl_program program = clCreateProgramWithSource(context, 1, (const char**)&source_str, (const size_t*)&source_size, &err);
	//std::cout << "clCreateProgramWithSource err: " << err << std::endl;
	err = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);
	//std::cout << "clBuildProgram err: " << err << std::endl;
	//cl_program_build_info build_info;
	//char* log = new char[10000];
	//clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, 10000 * sizeof(char), log, NULL);
	//std::cout << log << std::endl;
	
	auto t0 = omp_get_wtime();

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

#ifdef CHECKS
	size_t global_item_size = n;
#else
	size_t global_item_size = n < GLOBAL_SIZE ? n : GLOBAL_SIZE;
#endif // CHECKS
	size_t local_item_size = 0;
	err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global_item_size, NULL, 0, NULL, NULL);
	//std::cout << "clEnqueueNDRangeKernel err: " << err << std::endl;
	clFlush(queue);
	//clFinish(queue);

	clEnqueueReadBuffer(queue, input_output_y, CL_TRUE, 0, sizeof(float) * n, y, 0, NULL, NULL);

	auto t1 = omp_get_wtime();
	std::cout << "float ocl gpu t: " << t1 - t0 << std::endl;
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

void daxpy_gpu(int n, double a, double* x, int incx, double* y, int incy) {
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
	//std::cout << "clCreateContext err: " << err << std::endl;
	// print_device(context);
	cl_command_queue queue = clCreateCommandQueue(context, device_id, 0, &err);
	//std::cout << "clCreateCommandQueue err: " << err << std::endl;

	cl_mem input_x = clCreateBuffer(context, CL_MEM_READ_ONLY, n * sizeof(double), NULL, NULL);
	cl_mem input_output_y = clCreateBuffer(context, CL_MEM_READ_WRITE, n * sizeof(double), NULL, NULL);
	clEnqueueWriteBuffer(queue, input_x, CL_TRUE, 0, n * sizeof(double), x, 0, NULL, NULL);
	clEnqueueWriteBuffer(queue, input_output_y, CL_TRUE, 0, n * sizeof(double), y, 0, NULL, NULL);

	cl_program program = clCreateProgramWithSource(context, 1, (const char**)&source_str, (const size_t*)&source_size, &err);
	//std::cout << "clCreateProgramWithSource err: " << err << std::endl;
	err = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);
	//std::cout << "clBuildProgram err: " << err << std::endl;
	//cl_program_build_info build_info;
	//char* log = new char[10000];
	//clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, 10000 * sizeof(char), log, NULL);
	//std::cout << log << std::endl;

	auto t0 = omp_get_wtime();

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

#ifdef CHECKS
	size_t global_item_size = n;
#else
	size_t global_item_size = n < GLOBAL_SIZE ? n : GLOBAL_SIZE;
#endif // CHECKS

	size_t local_item_size = 0;
	err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global_item_size, NULL, 0, NULL, NULL);
	//std::cout << "clEnqueueNDRangeKernel err: " << err << std::endl;
	clFlush(queue);
	//clFinish(queue);

	clEnqueueReadBuffer(queue, input_output_y, CL_TRUE, 0, sizeof(double) * n, y, 0, NULL, NULL);

	auto t1 = omp_get_wtime();
	std::cout << "double ocl gpu t: " << t1 - t0 << std::endl;
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

void test_float(int N) {
	float* x_f = new float[N];
	float* y_f = new float[N];

	float* y_f_copy_1 = new float[N];
	float* y_f_copy_2 = new float[N];
	float* src = new float[N];

	float a_f = 2.5f;
	int incx = 2;
	int incy = 3;

	for (int i = 0; i < N; i++) {
		x_f[i] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX) * 123;
		y_f[i] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX) * 123;
		y_f_copy_1[i] = y_f[i];
		y_f_copy_2[i] = y_f[i];
		src[i] = y_f[i];
		#ifdef DEBUG
		std::cout << y_f[i] << " ";
		#endif // DEBUG
	}

	auto t0 = omp_get_wtime();
	saxpy(N, a_f, x_f, incx, y_f, incy);
	auto t1 = omp_get_wtime();
	std::cout << "float seq cpu t: " << t1 - t0 << std::endl;
	#ifdef DEBUG
	std::cout << '\n';
	for (int i = 0; i < N; i++) {
		std::cout << y_f[i] << " ";
	}
	std::cout << '\n';
	#endif // DEBUG

	t0 = omp_get_wtime();
	saxpy_omp(N, a_f, x_f, incx, y_f_copy_1, incy);
	t1 = omp_get_wtime();
	std::cout << "float omp cpu t: " << t1 - t0 << std::endl;
	#ifdef DEBUG
	for (int i = 0; i < N; i++) {
		std::cout << y_f_copy_1[i] << " ";
	}
	std::cout << '\n';
	#endif // DEBUG



	saxpy_gpu(N, a_f, x_f, incx, y_f_copy_2, incy);


#ifdef CHECKS
	for (int i = 0; i < N; i++) {
		// cpu float != gpu_float
		float _a = y_f[i];
		float _b = y_f_copy_1[i];
		float _c = y_f_copy_2[i];
		if (fabs(_a - _b) > EPS or fabs(_b - _c) > EPS or fabs(_c - _a) > EPS) {
			std::cout.precision(17);
			std::cout << "\nFail i=" << i << "; src=" << src[i] << ": " << _a << " " << _b << " " << _c << std::endl;
			//return;
		}
	}
#endif // CHECKS


	delete[] x_f;
	delete[] y_f;
	delete[] y_f_copy_1;
	delete[] y_f_copy_2;
}

void test_double(int N) {
	double* x_d = new double[N];
	double* y_d = new double[N];

	double* y_d_copy_1 = new double[N];
	double* y_d_copy_2 = new double[N];
	double* src = new double[N];

	double a_d = 2.5;
	int incx = 2;
	int incy = 3;

	for (int i = 0; i < N; i++) {
		x_d[i] = static_cast <double> (rand()) / static_cast <double> (RAND_MAX) * 123;
		y_d[i] = static_cast <double> (rand()) / static_cast <double> (RAND_MAX) * 123;
		y_d_copy_1[i] = y_d[i];
		y_d_copy_2[i] = y_d[i];
		src[i] = y_d[i];
#ifdef DEBUG
		std::cout << y_d[i] << " ";
#endif // DEBUG
	}
	std::cout << '\n';
	auto t0 = omp_get_wtime();
	daxpy(N, a_d, x_d, incx, y_d, incy);
	auto t1 = omp_get_wtime();
	std::cout << "double seq cpu t: " << t1 - t0 << std::endl;
#ifdef DEBUG
	std::cout << '\n';
	for (int i = 0; i < N; i++) {
		std::cout << y_d[i] << " ";
}
	std::cout << '\n';
#endif // DEBUG

	t0 = omp_get_wtime();
	daxpy_omp(N, a_d, x_d, incx, y_d_copy_1, incy);
	t1 = omp_get_wtime();
	std::cout << "double omp cpu t: " << t1 - t0 << std::endl;
#ifdef DEBUG
	for (int i = 0; i < N; i++) {
		std::cout << y_d_copy_1[i] << " ";
	}
	std::cout << '\n';
#endif // DEBUG



	daxpy_gpu(N, a_d, x_d, incx, y_d_copy_2, incy);


#ifdef CHECKS
	for (int i = 0; i < N; i++) {
		// cpu double != gpu_double
		double _a = y_d[i];
		double _b = y_d_copy_1[i];
		double _c = y_d_copy_2[i];
		std::cout.precision(17);
		if (fabs(_a - _b) > EPS or fabs(_b - _c) > EPS or fabs(_c - _a) > EPS) {
			std::cout << "\nFail i=" << i << "; src=" << src[i] << ": " << _a << " " << _b << " " << _c << std::endl;
			//return;
		}
	}
#endif // CHECKS

	delete[] x_d;
	delete[] y_d;
	delete[] y_d_copy_1;
	delete[] y_d_copy_2;
}




int main() {
	const int N = 100000000; // best 100000000, off checks
	print_info();
	test_float (N);
	test_double(N);
	return 0;
}