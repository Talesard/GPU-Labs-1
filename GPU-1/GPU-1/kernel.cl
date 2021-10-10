__kernel void sum(__global int* input, __global int* output, const int count) {
	int global_id = get_global_id(0);
	int group_id = get_group_id(0);
	int local_id = get_local_id(0);
	if (global_id < count) {
		output[global_id] = input[global_id] + global_id;
		printf("I am from %d block, %d thread (global index: %d)\n", local_id, group_id, global_id);
	}
}