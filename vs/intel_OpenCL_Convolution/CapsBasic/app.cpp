#include<stdio.h>
#include<stdlib.h>
#include<CL/cl.h>
#include<string.h>
#include <iostream>  

#define MAX_SOURCE_SIZE (0x10000)

int main(void)
{
	/*=================================================
	define parameters
	=================================================*/
	cl_platform_id	platform_id = NULL;
	cl_uint			ret_num_platforms;
	cl_device_id	device_id = NULL;
	cl_uint			ret_num_devices;
	cl_context		context = NULL;
	cl_command_queue command_queue = NULL;
	cl_mem			data_in = NULL;
	cl_mem			data_out = NULL;
	cl_mem			filter_in = NULL;
	cl_program		program = NULL;
	cl_kernel		kernel = NULL;
	size_t			kernel_code_size;
	char			*kernel_str;
	int				*result;
	cl_int			ret;
	FILE			*fp;
	cl_uint			work_dim;
	size_t			global_item_size[2];
	size_t			local_item_size[2];
	/*=================================================
	define parameters for image, filter, kernels
	=================================================*/
	int const W = 100;			//image width
	int const H = 100;			//image height
	int const K = 3;			//filter kernel size
	int const Wn = (W + K - 1); //padded image width
	int const Hn = (H + K - 1); //padded image height

	int point_num = Wn * Hn;
	int data_vecs[Wn*Hn];
	int filter_coex[K*K] = { -1,0,1,-2,0,2,-1,0,1 }; //sobel filter: horizontal gradient
	int filter_coey[K*K] = { -1,0,1,-2,0,2,-1,0,1 }; //sobel filter: horizontal gradient
	int i, j;

	for (i = 0; i < point_num; i++)
	{
		data_vecs[i] = rand() % 20;
	}

	//display input data
	printf("\n");
	printf("Array data_in:\n");
	for (i = 0; i < Hn; i++) {
		if(i<10)printf("row[%d]:\t", i);
		for (j = 0; j < Wn; j++) {
			if(j<10)printf("%d,\t", data_vecs[i*Wn + j]);
		}
		printf("\n");
	}
	printf("\n");


	/*=================================================
	load kernel, opencl environment setup
	create input and output buffer
	set kernel arguments, excute kernels
	get final results
	=================================================*/
	kernel_str = (char *)malloc(MAX_SOURCE_SIZE);
	result = (int *)malloc(W*H * sizeof(int));
	//遍历系统中所有OpenCL平台
	ret = clGetPlatformIDs(0, 0, &ret_num_platforms);
	printf("Num_of_platforms: %d\n", ret_num_platforms);
	cl_platform_id* platforms = new cl_platform_id[ret_num_platforms];
	ret = clGetPlatformIDs(ret_num_platforms, platforms, 0);
	cl_uint selected_platform_index = ret_num_platforms;
	for (cl_uint i = 0; i < ret_num_platforms; ++i)
	{
		size_t platform_name_length = 0;
		size_t maxComputeUnits = 0;
		ret = clGetPlatformInfo(platforms[i],CL_PLATFORM_NAME,0,0,&platform_name_length);
		// 调用两次，第一次是得到名称的长度
		char* platform_name = new char[platform_name_length];
		ret = clGetPlatformInfo(platforms[i],CL_PLATFORM_NAME,platform_name_length,platform_name,0);
		printf("%d:%s", i, platform_name);
		ret = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_GPU, 1, &device_id, NULL);
		ret = clGetDeviceInfo(device_id, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cl_uint), &maxComputeUnits, NULL);
		printf(" [PE=%d]", maxComputeUnits);
		if (strstr(platform_name, "Intel") && selected_platform_index == ret_num_platforms)
		{
			printf(" [Selected]");
			selected_platform_index = i;
		}
		printf("\n");
		delete[] platform_name;
	}
	if (selected_platform_index == ret_num_platforms)
	{
		printf("Intel platforms NOT FOUND!!\n");
		return 1;
	}

	cl_platform_id platform = platforms[selected_platform_index];
	ret = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device_id, NULL);
	context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &ret);
	command_queue = clCreateCommandQueue(context, device_id, 0, &ret);
	fp = fopen("kernelfun.cl", "r");
	kernel_code_size = fread(kernel_str, 1, MAX_SOURCE_SIZE, fp);
	fclose(fp);
	program = clCreateProgramWithSource(context, 1, (const char **)&kernel_str, (const size_t *)&kernel_code_size, &ret);
	ret = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);
#if 0
	//Shows the log
	char* build_log;
	size_t log_size;
	//First call to know the proper size
	clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
	build_log = new char[log_size + 1];
	// Second call to get the log
	clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, log_size, build_log, NULL);
	build_log[log_size] = '\0';
	printf("----------------\n");
	printf("build_log:%s", build_log);
	printf("----------------\n");
	delete[] build_log;
#endif
	kernel = clCreateKernel(program, "Conv2D", &ret);
	data_in = clCreateBuffer(context, CL_MEM_READ_WRITE, Wn*Hn * sizeof(int), NULL, &ret);
	data_out = clCreateBuffer(context, CL_MEM_READ_WRITE, W*H * sizeof(int), NULL, &ret);
	filter_in = clCreateBuffer(context, CL_MEM_READ_WRITE, K*K * sizeof(int), NULL, &ret);

	//write image data into data_in buffer
	ret = clEnqueueWriteBuffer(command_queue, data_in, CL_TRUE, 0, Wn*Hn * sizeof(int), data_vecs, 0, NULL, NULL);
	//write filter data into filter_in buffer
	ret = clEnqueueWriteBuffer(command_queue, filter_in, CL_TRUE, 0, K*K * sizeof(int), filter_coex, 0, NULL, NULL);

	//set kernel arguments
	ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&data_in);
	ret = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&filter_in);
	ret = clSetKernelArg(kernel, 2, sizeof(int), (void *)&K);
	ret = clSetKernelArg(kernel, 3, sizeof(cl_mem), (void *)&data_out);

	work_dim = 2;
	global_item_size[0] = { W };
	global_item_size[1] = { H };
	local_item_size[0] = { 1 };
	local_item_size[1] = { 1 };
	cl_event ev;
	//execute data parallel kernel */
	ret = clEnqueueNDRangeKernel(command_queue, kernel, work_dim, NULL,
			global_item_size, local_item_size, 0, NULL, &ev);
	clFinish(command_queue);
	//计算kerenl执行时间 
	cl_ulong startTime, endTime;
	clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_START,
		sizeof(cl_ulong), &startTime, NULL);
	clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_END,
		sizeof(cl_ulong), &endTime, NULL);
	cl_ulong kernelExecTimeNs = endTime - startTime;
	printf("GPU_1 running:%8.6f ms\n", kernelExecTimeNs*1e-6);
	// read data_out to host
	ret = clEnqueueReadBuffer(command_queue, data_out, CL_TRUE, 0,
			W*H * sizeof(int), result, 0, NULL, NULL);

	//display output data
	FILE *f_img_out = fopen("image_out.txt", "w+");
	printf("Array data_out: \n");
	for (i = 0; i < H; i++) {
		if (i<10)printf("row[%d]:\t", i);
		for (j = 0; j < W; j++) {
			if(j<10)printf("%d,\t", result[i*W + j]);
			fprintf(f_img_out, "%d,\t", result[i*W + j]);
		}
		printf("\n");
		fprintf(f_img_out, "\n");
	}
	printf("\n");
	fclose(f_img_out);
	/*=================================================
	release all opencl objects
	=================================================*/
	ret = clReleaseKernel(kernel);
	ret = clReleaseProgram(program);
	ret = clReleaseMemObject(data_in);
	ret = clReleaseMemObject(data_out);
	ret = clReleaseMemObject(filter_in);
	ret = clReleaseCommandQueue(command_queue);
	ret = clReleaseContext(context);
	free(result);
	free(kernel_str);

	system("pause");
	return 0;
}
