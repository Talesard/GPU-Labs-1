__kernel void mult_classic(__global float* A, __global float* B, __global float* Res, int m, int n, int k) {
    int r = get_global_id(1);
    int c = get_global_id(0);
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        sum += A[r * n + i] * B[c + n * i];
    }
    Res[k * r + c] = sum;
}






__kernel void mult_block(__global float *A, __global float *B, __global float *Res, int m, int n, int k) {
    __local float _A[16][16];
    __local float _B[16][16];

    int block_r = get_local_id(0);
    int block_c = get_local_id(1);

    int global_r = get_global_id(0);
    int global_c = get_global_id(1);

    int block_count = m / 16;
    float sum = 0.0f;

    for (int i = 0; i < block_count; i++) {

        _A[block_c][block_r] = A[global_c * m + 16 * i + block_r];
        _B[block_c][block_r] = B[16 * i * n + block_c * n + global_r];

        barrier(CLK_LOCAL_MEM_FENCE);
        for (int j = 0; j < 16; j++) {
            sum += _A[block_c][j] * _B[j][block_r];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    Res[global_c * k + global_r] = sum;
}







__kernel void mult_image(__read_only image2d_t A, __read_only image2d_t B, __write_only image2d_t Res, int m, int n, int k) {
    __local float _A[16][16];
    __local float _B[16][16];

    int block_r = get_local_id(0);
    int block_c = get_local_id(1);

    int global_r = get_global_id(0);
    int global_c = get_global_id(1);

    int block_count = m / 16;
    float sum = 0.0f;

    for (int i = 0; i < block_count; i++) {

        // read_image для типа картинки CL_R -> (r, 0.0, 0.0, 1.0) => .x чтобы взять значение

        _A[block_c][block_r] = read_imagef(A, (int2)(16 * i + block_r, global_c)).x;
        _B[block_c][block_r] = read_imagef(B, (int2)(global_r, 16 * i + block_c)).x;

        barrier(CLK_LOCAL_MEM_FENCE);
        for (int j = 0; j < 16; j++) {
            sum += _A[block_c][j] * _B[j][block_r];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    write_imagef(Res, (int2)(global_r, global_c), sum);
}