#pragma once
#include <vector>
#include <CL/cl.h>
#include <iostream>
#define MAX_SOURCE_SIZE (0x100000)

std::pair<char*, size_t> read_kernel(const char* filename) {
	FILE* fp;
	char* source_str;
	size_t source_size;
	fp = fopen(filename, "r");
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