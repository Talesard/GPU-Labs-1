#pragma once
#include <omp.h>

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