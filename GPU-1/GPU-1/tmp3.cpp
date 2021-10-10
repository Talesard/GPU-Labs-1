//#include <CL/cl.h>
//#include <iostream>
//
//constexpr auto GPU = 0;
//constexpr auto CPU = 1;
//
//std::pair<cl_platform_id*, cl_uint> get_info() {
//	// сначала получаем количество платформ
//	cl_uint platformCount = 0;
//	clGetPlatformIDs(0, nullptr, &platformCount);
//	// теперь получаем сами платформы
//	cl_platform_id* platforms = new cl_platform_id[platformCount];
//	clGetPlatformIDs(platformCount, platforms, nullptr);
//	// выводим все доступные платформы
//	for (cl_uint i = 0; i < platformCount; ++i) {
//		char platformName[128];
//		clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME,
//			128, platformName, nullptr);
//		std::cout << "[" << i << "] " << platformName << std::endl;
//
//		cl_context_properties properties[3] = { CL_CONTEXT_PLATFORM, (cl_context_properties)platforms[i], 0 };
//		cl_context context = NULL;
//		if (i == 0) {
//			context = clCreateContextFromType((NULL == platforms[i]) ? NULL : properties, CL_DEVICE_TYPE_GPU, NULL, NULL, NULL);
//		}
//		else {
//			context = clCreateContextFromType((NULL == platforms[i]) ? NULL : properties, CL_DEVICE_TYPE_CPU, NULL, NULL, NULL);
//		}
//		size_t size = 0;
//		clGetContextInfo(context, CL_CONTEXT_DEVICES, 0, NULL, &size);
//		cl_device_id* devices = (cl_device_id*)_malloca(size);
//		clGetContextInfo(context, CL_CONTEXT_DEVICES, size, devices, NULL);
//		cl_device_id device = NULL;
//		device = devices[0];
//		char deviceName[100];
//		size_t real_size{};
//		clGetDeviceInfo(device, CL_DEVICE_NAME, 100, deviceName, &real_size);
//		std::cout << "   " << "[" << 0 << "] " << deviceName << std::endl;
//	}
//	std::cout << "------------------------------------------------" << std::endl;
//	return std::pair<cl_platform_id*, cl_uint>{platforms, platformCount};
//}
//
//std::pair<cl_device_id, cl_context> get_device_and_context(cl_platform_id* platforms, cl_uint platformCount, cl_uint gpu_or_cpu) {
//	cl_platform_id platform = NULL;
//	if (platformCount > 1) {
//		platform = platforms[gpu_or_cpu];
//	}
//
//	// свойства контекста
//	cl_context_properties properties[3] = { CL_CONTEXT_PLATFORM, (cl_context_properties)platform, 0 };
//
//	// контекст для гпу
//	cl_context context = clCreateContextFromType(
//		(NULL == platform) ? NULL : properties, gpu_or_cpu == 0 ? CL_DEVICE_TYPE_GPU : CL_DEVICE_TYPE_CPU, NULL, NULL, NULL);
//
//	// размер массива в байтах для хранения списка устройств.
//	size_t size = 0;
//	clGetContextInfo(context, CL_CONTEXT_DEVICES, 0, NULL, &size);
//
//	// выбираем девайс
//	cl_device_id device = NULL;
//	if (size > 0) {
//		cl_device_id* devices = (cl_device_id*)_malloca(size); // alloca
//		clGetContextInfo(context, CL_CONTEXT_DEVICES, size, devices, NULL);
//		//у меня одна карточка и один проц, значит выбираем нулевой девайс
//		device = devices[0];
//		char deviceName[100];
//		size_t real_size{};
//		clGetDeviceInfo(device, CL_DEVICE_NAME, 100, deviceName, &real_size);
//		std::cout << "Using: " << deviceName << std::endl;
//	}
//	return std::pair<cl_device_id, cl_context>{device, context};
//}
//
//int main() {
//	auto info = get_info();
//	auto platforms = info.first;
//	auto platformCount = info.second;
//
//	auto device_context = get_device_and_context(platforms, platformCount, GPU);
//	auto device = device_context.first;
//	auto context = device_context.second;
//
//	// создаем очередь команд для контекста и гпу
//	cl_command_queue queue = clCreateCommandQueue(context, device, 0, NULL);
//	// сорс гпу проги
//	const char* source =
//		"__kernel void square() {\n"\
//		"constant char *p = \"this is a test string\";"
//		" printf(\"%s\", p);\n"\
//		"}";
//	//"__kernel void square(__global float* input, __global float* output, const unsigned int count) {\n"\
//		//"	int i = get_global_id(0);\n"\
//		//"	if (i < count) {\n"\
//		//"		output[i] = input[i] * input[i];\n"\
//		//"		printf(\"123\");\n"\
//		//"	}\n"\
//		//"}";
//
//	// собираем гпу прогу
//	size_t srclen[] = { strlen(source) };
//	cl_int err;
//	cl_program program = clCreateProgramWithSource(context, 1, &source, srclen, &err);
//	std::cout << "err: " << err << std::endl;
//	clBuildProgram(program, 1, &device, NULL, NULL, NULL);
//	cl_program_build_info build_info;
//	char* log = new char[10000];
//	clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 10000 * sizeof(char), log, NULL);
//	std::cout << log << std::endl;
//
//	// объект ядра для проги
//	cl_kernel kernel = clCreateKernel(program, "square", &err);
//	std::cout << "err: " << err << std::endl;
//
//	// входные и выходные данные
//	const int SIZE = 100;
//	float* data = new float[SIZE];
//	float* results = new float[SIZE];
//	for (int i = 0; i < SIZE; i++) {
//		data[i] = rand();
//		results[i] = -1;
//		// std::cout << data[i] << std::endl;
//	}
//
//	//// входной буффер
//	//cl_mem input = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float)*SIZE, NULL, NULL);
//
//	//// выходной буффер
//	//cl_mem output = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * SIZE, NULL, NULL);
//
//	//// копируем входной буффер на гпу
//	//clEnqueueWriteBuffer(queue, input, CL_TRUE, 0, sizeof(float) * SIZE, data, 0, NULL, NULL);
//
//	// аргументы ядра
//	size_t count = SIZE;
//	//clSetKernelArg(kernel, 0, sizeof(cl_mem), &input);
//	//clSetKernelArg(kernel, 1, sizeof(cl_mem), &output);
//	//clSetKernelArg(kernel, 2, sizeof(cl_mem), &count);
//
//	// группа работ
//	size_t group;
//	clGetKernelWorkGroupInfo(kernel, device, CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), &group, NULL);
//
//	// запуск ядра
//	std::cout << "start" << std::endl;
//	clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &count, &group, 0, NULL, NULL);
//	std::cout << "end" << std::endl;
//	clFlush(queue);
//	clFinish(queue);
//
//	// загрузка результатов из гпу
//	// clEnqueueReadBuffer(queue, output, CL_TRUE, 0, sizeof(float) * count, results, 0, NULL, NULL);
//
//	// освобождение ресурсов
//	// clReleaseMemObject(input);
//	// clReleaseMemObject(output);
//	clReleaseProgram(program);
//	clReleaseKernel(kernel);
//	clReleaseCommandQueue(queue);
//	clReleaseContext(context);
//
//	// вывод результатов
//	//for (int i = 0; i < SIZE; i++) {
//	//	std::cout << data[i] << "^2= " << results[i] << std::endl;
//	//} 
//
//	return 0;
//}