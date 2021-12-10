#define _CRT_SECURE_NO_WARNINGS
#include <iostream>
#include <omp.h>
#include <CL/cl.h>
#include <random>
#include "ocl_helpers.h"
#include "matrix_mult.h"
#define EPS 1e-3

float* get_random_matrix(int m, int n) {
	std::random_device dev;
	std::mt19937 rng(dev());
	std::uniform_int_distribution<std::mt19937::result_type> distr(0, 9);
	float* res = new float[m * n];
	for (int i = 0; i < m * n; i++) {
		res[i] = static_cast <float> (distr(rng));
	}
	return res;
}

bool check_res(float* Res1, float* Res2, float size) {
	for (int i = 0; i < size; i++) {
		if (abs(Res1[i] - Res2[i]) > EPS) {
			std::cout << "Check fail: " << Res1[i] << " != " << Res2[i] << std::endl;
			return false;
		}
	}
	return true;
}

void fill_zero(float* A, int size) {
#pragma omp parallel for
	for (int i = 0; i < size; i++) {
		A[i] = 0;
	}
}

void print_matrix(float* A, int m, int n) {
	for (int i = 0; i < m * n; i++) {
		if (i % n == 0) std::cout << std::endl;
		std::cout << A[i] << " ";
	}
}

int main() {
	print_info();

	float t0;
	float t1;

	const int m = 1600; //1600
	const int n = 1600;
	const int k = 1600;

	float* A = get_random_matrix(m, n);
	float* B = get_random_matrix(n, k);

	float* Res_classic_cpu = new float[m * k];
	fill_zero(Res_classic_cpu, m * k);
	float* Res_classic_cpu_omp = new float[m * k];
	fill_zero(Res_classic_cpu_omp, m * k);

	float* Res_classic_cpu_cl = new float[m * k];
	fill_zero(Res_classic_cpu_cl, m * k);
	float* Res_classic_gpu_cl = new float[m * k];
	fill_zero(Res_classic_gpu_cl, m * k);

	float* Res_block_gpu_cl = new float[m * k];
	fill_zero(Res_block_gpu_cl, m * k);
	float* Res_block_cpu_cl = new float[m * k];
	fill_zero(Res_block_cpu_cl, m * k);

	float* Res_image_gpu_cl = new float[m * k];
	fill_zero(Res_image_gpu_cl, m * k);
	float* Res_image_cpu_cl = new float[m * k];
	fill_zero(Res_image_cpu_cl, m * k);

	t0 = omp_get_wtime();
	mult_classic_cpu(A, B, Res_classic_cpu, m, n, k);
	t1 = omp_get_wtime();
	std::cout << "Time classic CPU seq: " << t1 - t0 << std::endl;

	t0 = omp_get_wtime();
	mult_classic_cpu_omp(A, B, Res_classic_cpu_omp, m, n, k);
	t1 = omp_get_wtime();
	std::cout << "Time classic CPU omp: " << t1 - t0 << std::endl;
	
	std::cout << "Check classic omp: " << check_res(Res_classic_cpu, Res_classic_cpu_omp, m*k) << std::endl << std::endl;

	mult_classic_cl(A, B, Res_classic_cpu_cl, m, n, k, 1);
	std::cout << "Check classic cpu cl: " << check_res(Res_classic_cpu, Res_classic_cpu_cl, m * k) << std::endl << std::endl;

	mult_classic_cl(A, B, Res_classic_gpu_cl, m, n, k, 0);
	std::cout << "Check classic gpu cl: " << check_res(Res_classic_cpu, Res_classic_gpu_cl, m * k) << std::endl << std::endl;

	mult_block_cl(A, B, Res_block_cpu_cl, m, n, k, 1);
	std::cout << "Check block cpu cl: " << check_res(Res_classic_cpu, Res_block_cpu_cl, m * k) << std::endl << std::endl;

	mult_block_cl(A, B, Res_block_gpu_cl, m, n, k, 0);
	std::cout << "Check block gpu cl: " << check_res(Res_classic_cpu, Res_block_gpu_cl, m * k) << std::endl << std::endl;

	mult_image_cl(A, B, Res_image_cpu_cl, m, n, k, 1);
	std::cout << "Check image cpu cl: " << check_res(Res_classic_cpu, Res_image_cpu_cl, m * k) << std::endl << std::endl;

	mult_image_cl(A, B, Res_image_gpu_cl, m, n, k, 0);
	std::cout << "Check image gpu cl: " << check_res(Res_classic_cpu, Res_image_gpu_cl, m * k) << std::endl << std::endl;

	//// tmp
	//print_matrix(A, m, n);
	//std::cout << "\nx" << std::endl;
	//print_matrix(B, n, k);
	//std::cout << "\nx" << std::endl;
	//print_matrix(Res_classic_cpu, m, k);

	return 0;
}