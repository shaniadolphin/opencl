
__kernel void RunAsGpu_1(
	__global  float *A,
	__global  float *B,
	int M,
	int N,
	int P,
	__global float *C)
{
	int x = get_global_id(0);
	int y = get_global_id(1);
	float sum = 0;
	for(int i = 0;i<P;i++)
	{
		sum += A[x*P + i]*B[i*N + y];
	}
	C[x*N + y] = sum;
}

__kernel void RunAsGpu_2(
	__global  float *A,
	__global  float *B,
	int M,
	int N,
	int P,
	__global float *C)
{
	int x = get_global_id(0);
	int y = get_global_id(1);
	float sum = 0;
	for(int i = 0;i<P;i++)
	{
		sum += A[y*P + i]*B[i*N + x];
	}
	C[y*N + x] = sum;
}

__kernel void RunAsGpu_3(
	__global  float *A,
	__global  float *B,
	int M,
	int N,
	int P,
	__global float *C)
{
	int x = get_global_id(0);
	int y = get_global_id(1);
	C[x*N + y] = 0;
	for(int i = 0;i<P;i++)
	{
		C[x*N + y] += A[x*P + i]*B[i*N + y];
	}
}