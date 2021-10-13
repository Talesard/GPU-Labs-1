__kernel void saxpy(const int n, const float a, __global float* x, const int incx, __global float* y, const int incy) {
	int global_id = get_global_id(0);
	if (global_id < n) {
		int id_y = global_id * incy;
		int id_x = global_id * incx;
		if ((id_y >= 0 && id_y < n) && (id_x >= 0 && id_x < n))
			y[id_y] = y[id_y] + a * x[id_x];
	}
}

__kernel void daxpy(const int n, const double a, __global double* x, const int incx, __global double* y, const int incy) {
	int global_id = get_global_id(0);
	if (global_id < n) {
		int id_y = global_id * incy;
		int id_x = global_id * incx;
		if ((id_y >= 0 && id_y < n) && (id_x >= 0 && id_x < n))
			y[id_y] = y[id_y] + a * x[id_x];
	}
}