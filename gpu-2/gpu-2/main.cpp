#define _CRT_SECURE_NO_WARNINGS
#include <omp.h>
#include <iostream>
#include <vector>
#include <CL/cl.h>

#include "ocl_helpers.h"
#include "cpu_default.h"
#include "cpu_omp.h"
#include "gpu_cpu_ocl.h"

#define EPS 1e-3

void check_float(float* cpu_def, float* cpu_omp, float* ocl_cpu, float* ocl_gpu, int size) {
	for (int i = 0; i < size; i++) {

		if (!(fabs(cpu_def[i] - cpu_omp[i]) < EPS
			&& fabs(cpu_omp[i] - ocl_cpu[i]) < EPS
			&& fabs(ocl_cpu[i] - ocl_gpu[i]) < EPS
			&& fabs(ocl_gpu[i] - cpu_def[i]) < EPS)) {

			std::cout << "float check fail!" << std::endl;
			std::cout << "cpu_def: " << cpu_def[i] << std::endl;
			std::cout << "cpu_omp: " << cpu_omp[i] << std::endl;
			std::cout << "cpu_ocl: " << ocl_cpu[i] << std::endl;
			std::cout << "gpu_ocl: " << ocl_gpu[i] << std::endl;
			return;
		}
	}
	std::cout << "successful float check" << std::endl;
}

void check_double(double* cpu_def, double* cpu_omp, double* ocl_cpu, double* ocl_gpu, int size) {
	for (int i = 0; i < size; i++) {

		if (!(fabs(cpu_def[i] - cpu_omp[i]) < EPS
			&& fabs(cpu_omp[i] - ocl_cpu[i]) < EPS
			&& fabs(ocl_cpu[i] - ocl_gpu[i]) < EPS
			&& fabs(ocl_gpu[i] - cpu_def[i]) < EPS)) {

			std::cout << "double check fail!" << std::endl;
			std::cout << "cpu_def: " << cpu_def[i] << std::endl;
			std::cout << "cpu_omp: " << cpu_omp[i] << std::endl;
			std::cout << "cpu_ocl: " << ocl_cpu[i] << std::endl;
			std::cout << "gpu_ocl: " << ocl_gpu[i] << std::endl;
			return;
		}
	}
	std::cout << "successful double check" << std::endl;
}

void test_float(int N, size_t local_sz) {
	float* x_f = new float[N];
	float* y_f = new float[N];

	float* y_f_copy_1 = new float[N];
	float* y_f_copy_2 = new float[N];
	float* y_f_copy_3 = new float[N];
	float* src = new float[N];

	float a_f = 2.5f;
	int incx = 2;
	int incy = 3;

	for (int i = 0; i < N; i++) {
		x_f[i] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX) * 123;
		y_f[i] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX) * 123;
		y_f_copy_1[i] = y_f[i];
		y_f_copy_2[i] = y_f[i];
		y_f_copy_3[i] = y_f[i];
		src[i] = y_f[i];
		// std::cout << y_f[i] << " ";
	}

	
	// CPU default
	auto t0 = omp_get_wtime();
	saxpy(N, a_f, x_f, incx, y_f, incy);
	auto t1 = omp_get_wtime();
	std::cout << "float CPU default: " << t1 - t0 << std::endl;

	// CPU omp
	t0 = omp_get_wtime();
	saxpy_omp(N, a_f, x_f, incx, y_f_copy_1, incy);
	t1 = omp_get_wtime();
	std::cout << "float CPU omp: " << t1 - t0 << std::endl;

	// CPU ocl
	t0 = omp_get_wtime();
	saxpy_gpu(N, a_f, x_f, incx, y_f_copy_2, incy, 1, local_sz);
	t1 = omp_get_wtime();
	std::cout << "float CPU ocl all: " << t1 - t0 << std::endl;

	// GPU ocl
	t0 = omp_get_wtime();
	saxpy_gpu(N, a_f, x_f, incx, y_f_copy_3, incy, 0, local_sz);
	t1 = omp_get_wtime();
	std::cout << "float GPU ocl all: " << t1 - t0 << std::endl;

	check_float(y_f, y_f_copy_1, y_f_copy_2, y_f_copy_3, N);

	delete[] x_f;
	delete[] y_f;
	delete[] y_f_copy_1;
	delete[] y_f_copy_2;
	delete[] y_f_copy_3;
}

void test_double(int N, size_t local_sz) {
	double* x_f = new double[N];
	double* y_f = new double[N];

	double* y_f_copy_1 = new double[N];
	double* y_f_copy_2 = new double[N];
	double* y_f_copy_3 = new double[N];
	double* src = new double[N];

	double a_f = 2.5;
	int incx = 2;
	int incy = 3;

	for (int i = 0; i < N; i++) {
		x_f[i] = static_cast <double> (rand()) / static_cast <double> (RAND_MAX) * 123;
		y_f[i] = static_cast <double> (rand()) / static_cast <double> (RAND_MAX) * 123;
		y_f_copy_1[i] = y_f[i];
		y_f_copy_2[i] = y_f[i];
		y_f_copy_3[i] = y_f[i];
		src[i] = y_f[i];
		// std::cout << y_f[i] << " ";
	}


	// CPU default
	auto t0 = omp_get_wtime();
	daxpy(N, a_f, x_f, incx, y_f, incy);
	auto t1 = omp_get_wtime();
	std::cout << "double CPU default: " << t1 - t0 << std::endl;

	// CPU omp
	t0 = omp_get_wtime();
	daxpy_omp(N, a_f, x_f, incx, y_f_copy_1, incy);
	t1 = omp_get_wtime();
	std::cout << "double CPU omp: " << t1 - t0 << std::endl;

	// CPU ocl
	t0 = omp_get_wtime();
	daxpy_gpu(N, a_f, x_f, incx, y_f_copy_2, incy, 1, local_sz);
	t1 = omp_get_wtime();
	std::cout << "double CPU ocl all: " << t1 - t0 << std::endl;

	// GPU ocl
	t0 = omp_get_wtime();
	daxpy_gpu(N, a_f, x_f, incx, y_f_copy_3, incy, 0, local_sz);
	t1 = omp_get_wtime();
	std::cout << "double GPU ocl all: " << t1 - t0 << std::endl;

	check_double(y_f, y_f_copy_1, y_f_copy_2, y_f_copy_3, N);

	delete[] x_f;
	delete[] y_f;
	delete[] y_f_copy_1;
	delete[] y_f_copy_2;
	delete[] y_f_copy_3;
}




int main() {
	const int N = 256 * 450000;
	size_t local_size = 256;
	print_info();

	test_float(N, local_size);
	std::cout << "\n\n" << std::endl;
	test_double(N, local_size);

	return 0;
}