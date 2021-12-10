__kernel void mult_classic(__global float* A, __global float* B, __global float* Res, int m, int n, int k) {
    /*
    * матрица хранитс€ линейно, строки идут друг за другом
    * i от потока не зависит
    * доступы оптимальные, лучше не возможно
    * доступ к A с шагом n
    * доступ к B линейный
    * 
    * get_global_id(0) дает изменение потоков 0 1 2 ... последовательно
    * get_global_id(1) дает изменение потоков с шагов в размер группы
    * row=get_global_id(...) умножаетс€ на n в ј[], поэтому ему ничего не поможет, все равно будет шаг
    * col=get_global_id(...) не умножаетс€, поэтому если поставить 0, то доступа с шагом не будет.
    */
    int r = get_global_id(1);
    int c = get_global_id(0);
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        sum += A[r * n + i] * B[c + n * i];
    }
    Res[k * r + c] = sum;
}


__kernel void mult_block(__global float *A, __global float *B, __global float *Res, int m, int n, int k) {
    // __local доступен в группе
    __local float _A[16][16];
    __local float _B[16][16];

    int block_r = get_local_id(0); // номер потока внутри группы
    int block_c = get_local_id(1); // номер потока внутри группы

    int global_r = get_global_id(0); // глобальный номер потока
    int global_c = get_global_id(1); // глобальный номер потока

    int block_count = m / 16; // общее число блоков в матрице (вернее не общее, а число блоков по одной оси)
    float sum = 0.0f;

    for (int i = 0; i < block_count; i++) {
        // global_c*m - дойти до начала строки потока(глобал)                       ???
        // +16*i дошли до начала нужного блока                                      ???
        // +block_r вз€ли элемент блока соответствующий номеру потока(локал)        ???

        /*
        * ƒоступ к ј: global_c*m в любом случае с шагом M, поэтому global_c = get_global_id(0),
        * который идет без шага в размер группы по оси не поможет.
        * 
        * ƒоступ к ¬: global_r без шага, поэтому  global_r=get_global_id(0), который без шага оптимальнее
        */
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

        // read_image for CL_R -> (r, 0.0, 0.0, 1.0) => .x чтобы вз€ть значение

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