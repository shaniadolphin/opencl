#include <iostream>
#include <CL/cl.h>
#include <cassert>
#include <windows.h>
#include <ctime>
using namespace std;


#define M 800
#define P 500
#define N 800

void RunAsCpu(
	const float *A,
	const float *B,
	float* C)
{
	for (int i = 0; i < M; i++)
	{
		for (int j = 0; j < N; j++)
		{
			C[i*N + j] = 0.0;
			for (int k = 0; k < P; k++)
			{
				C[i*N + j] += A[i*P + k] * B[k*N + j];
			}
		}
	}
}

//计时函数
double time_stamp()
{
	LARGE_INTEGER curclock;
	LARGE_INTEGER freq;
	if (
		!QueryPerformanceCounter(&curclock) ||
		!QueryPerformanceFrequency(&freq)
		)
	{
		return -1;
	}

	return double(curclock.QuadPart) / freq.QuadPart;
}
#define OPENCL_CHECK_ERRORS(ERR)        \
    if(ERR != CL_SUCCESS)                  \
    {                                      \
    cerr                                   \
    << "OpenCL error with code " << ERR    \
    << " happened in file " << __FILE__    \
    << " at line " << __LINE__             \
    << ". Exiting...\n";                   \
    exit(1);                               \
    }
int main(int argc, const char** argv)
{
	cl_int error = 0;   // Used to handle error codes
	cl_context context;
	cl_command_queue queue;
	cl_device_id device;

	//遍历系统中所有OpenCL平台
	cl_uint num_of_platforms = 0;
	//得到平台数目
	error = clGetPlatformIDs(0, 0, &num_of_platforms);
	OPENCL_CHECK_ERRORS(error);
	//cout << "Num_of_platforms: " << num_of_platforms << endl;
	printf("Num_of_platforms: %d\n", num_of_platforms);
	cl_platform_id* platforms = new cl_platform_id[num_of_platforms];
	//得到所有平台的ID
	error = clGetPlatformIDs(num_of_platforms, platforms, 0);
	OPENCL_CHECK_ERRORS(error);
	//遍历平台，选择一个Intel平台的
	cl_uint selected_platform_index = num_of_platforms;
	for (cl_uint i = 0; i < num_of_platforms; ++i)
	{
		size_t platform_name_length = 0;
		error = clGetPlatformInfo(
			platforms[i],
			CL_PLATFORM_NAME,
			0,
			0,
			&platform_name_length
		);
		OPENCL_CHECK_ERRORS(error);

		// 调用两次，第一次是得到名称的长度
		char* platform_name = new char[platform_name_length];
		error = clGetPlatformInfo(
			platforms[i],
			CL_PLATFORM_NAME,
			platform_name_length,
			platform_name,
			0
		);
		OPENCL_CHECK_ERRORS(error);

		printf("%d:%s",i, platform_name);
		if (
			strstr(platform_name, "Intel") &&
			selected_platform_index == num_of_platforms // have not selected yet
			)
		{
			printf(" [Selected]");
			selected_platform_index = i;
		}
		printf("\n");
		delete[] platform_name;
	}
	if (selected_platform_index == num_of_platforms)
	{
		printf("Intel platforms NOT FOUND!!\n");
		return 1;
	}
	//Device
	cl_platform_id platform = platforms[selected_platform_index];
	error = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
	OPENCL_CHECK_ERRORS(error)

	//Context
	context = clCreateContext(0, 1, &device, NULL, NULL, &error);
	OPENCL_CHECK_ERRORS(error)

	// Command-queue CL_QUEUE_PROFILING_ENABLE开启才能计时
	queue = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &error);
	OPENCL_CHECK_ERRORS(error)

	//下面初始化测试数据(主机数据)
	float* A_h = new float[M*P];
	float* B_h = new float[P*N];
	float* C_h = new float[M*N];
	srand(100);
	for (int i = 0; i < M*P; i++)
		A_h[i] = (float)(rand() % 50);

	for (int i = 0; i < P*N; i++)
		B_h[i] = (float)(rand() % 50);
	//初始化设备数据
	// 标志位表示数据只读，并且从nums1_h和nums2_h复制数据
	cl_mem A_d = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float)*M*P, A_h, &error);
	OPENCL_CHECK_ERRORS(error)
		cl_mem B_d = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float)*P*N, B_h, &error);
	OPENCL_CHECK_ERRORS(error)
		cl_mem C_d = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float)*M*N, NULL, &error);
	OPENCL_CHECK_ERRORS(error)

	double starttime = time_stamp();
	RunAsCpu(A_h, B_h, C_h);
	double stoptime = time_stamp();
	printf("CPU running:%8.6f s\n", stoptime - starttime);

	//读取OpenCLSum.cl文件内容
	FILE* fp = fopen("OpenCLMulMatrix.cl", "rb");
	fseek(fp, 0, SEEK_END);
	size_t src_size = ftell(fp);
	fseek(fp, 0, SEEK_SET);
	const char* source = new char[src_size];
	fread((void*)source, 1, src_size, fp);
	fclose(fp);

	//创建编译运行kernel函数
	cl_program program = clCreateProgramWithSource(context, 1, &source, &src_size, &error);
	OPENCL_CHECK_ERRORS(error)
	delete[] source;

	// Builds the program
	error = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
	OPENCL_CHECK_ERRORS(error)

	//Shows the log
	char* build_log;
	size_t log_size;
	//First call to know the proper size
	clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
	build_log = new char[log_size + 1];
	// Second call to get the log
	clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, build_log, NULL);
	build_log[log_size] = '\0';
	printf("----------------\n");
	printf("build_log:%s", build_log);
	printf("----------------\n");
	//cout << build_log << endl;
	delete[] build_log;

	//Extracting the kernel
	cl_kernel run_as_gpu_1 = clCreateKernel(program, "RunAsGpu_2", &error);
	OPENCL_CHECK_ERRORS(error)
	//设置kernel参数
	cl_int M_d = M;
	cl_int P_d = P;
	cl_int N_d = N;
	error = clSetKernelArg(run_as_gpu_1, 0, sizeof(cl_mem), &A_d);
	error |= clSetKernelArg(run_as_gpu_1, 1, sizeof(cl_mem), &B_d);
	error |= clSetKernelArg(run_as_gpu_1, 2, sizeof(int), &M_d);
	error |= clSetKernelArg(run_as_gpu_1, 3, sizeof(int), &N_d);
	error |= clSetKernelArg(run_as_gpu_1, 4, sizeof(int), &P_d);
	error |= clSetKernelArg(run_as_gpu_1, 5, sizeof(cl_mem), &C_d);
	OPENCL_CHECK_ERRORS(error)
	//启动kernel
	size_t globalws_1[2] = { M,N };
	cl_event ev;
	error = clEnqueueNDRangeKernel(queue, run_as_gpu_1, 2, NULL, globalws_1, NULL, 0, NULL, &ev);
	clFinish(queue);
	OPENCL_CHECK_ERRORS(error)
	//计算kerenl执行时间 
	cl_ulong startTime, endTime;
	clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_START,
		sizeof(cl_ulong), &startTime, NULL);
	clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_END,
		sizeof(cl_ulong), &endTime, NULL);
	cl_ulong kernelExecTimeNs = endTime - startTime;
	printf("GPU_1 running:%8.6f ms\n", kernelExecTimeNs*1e-6);
	//取得kernel返回值
	float* gpu_C_1 = new float[M*N];
	clEnqueueReadBuffer(queue, C_d, CL_TRUE, 0, M*N*sizeof(float), gpu_C_1, 0, NULL, NULL);
	assert(memcmp(C_h, gpu_C_1, M*N * sizeof(float)) == 0);


	//Extracting the kernel
	cl_kernel run_as_gpu_2 = clCreateKernel(program, "RunAsGpu_2", &error);
	OPENCL_CHECK_ERRORS(error)
	//设置kernel参数
	error = clSetKernelArg(run_as_gpu_2, 0, sizeof(cl_mem), &A_d);
	error |= clSetKernelArg(run_as_gpu_2, 1, sizeof(cl_mem), &B_d);
	error |= clSetKernelArg(run_as_gpu_2, 2, sizeof(int), &M_d);
	error |= clSetKernelArg(run_as_gpu_2, 3, sizeof(int), &N_d);
	error |= clSetKernelArg(run_as_gpu_2, 4, sizeof(int), &P_d);
	error |= clSetKernelArg(run_as_gpu_2, 5, sizeof(cl_mem), &C_d);
	OPENCL_CHECK_ERRORS(error)

	// 启动kernel
	size_t globalws_2[2] = { N,M };
	error = clEnqueueNDRangeKernel(queue, run_as_gpu_2, 2, NULL, globalws_2, NULL, 0, NULL, &ev);
	clFinish(queue);
	OPENCL_CHECK_ERRORS(error)
	//计算kerenl执行时间 
	clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_START,
		sizeof(cl_ulong), &startTime, NULL);
	clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_END,
		sizeof(cl_ulong), &endTime, NULL);
	kernelExecTimeNs = endTime - startTime;
	printf("GPU_2 running:%8.6f ms\n", kernelExecTimeNs*1e-6);
	//取得kernel返回值
	float* gpu_C_2 = new float[M*N];
	clEnqueueReadBuffer(queue, C_d, CL_TRUE, 0, M*N * sizeof(float), gpu_C_2, 0, NULL, NULL);
	assert(memcmp(C_h, gpu_C_2, M*N * sizeof(float)) == 0);

	delete[] A_h;
	delete[] B_h;
	delete[] C_h;
	delete[] gpu_C_1;
	delete[] gpu_C_2;
	delete[] platforms;
	clReleaseKernel(run_as_gpu_1);
	clReleaseKernel(run_as_gpu_2);
	clReleaseCommandQueue(queue);
	clReleaseContext(context);
	clReleaseMemObject(A_d);
	clReleaseMemObject(B_d);
	clReleaseMemObject(C_d);
	return 0;
}