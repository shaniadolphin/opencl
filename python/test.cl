__kernel void test(
	__global float *a, 
	__global float *b, 
	const float c) 
{
  size_t i = get_global_id(0);
  a[i] = (a[i] + b[i]) * c;
}

__kernel __attribute__((vec_type_hint(float4)))void matmul(
	__global const float *a,
	__global const float *b,
	__global float *c,
	const int m,
	const int p,
	const int n)
{
	const int x = get_global_id(0);
	const int y = get_global_id(1);
	float sum = 0;
	for(size_t i = 0;i<p;i++)
	{
		sum += a[y*p + i] * b[i*n + x];
		
	}
	c[y*n + x] = sum;
}

__kernel void testadd(
	__global const float *a,
	__global const float *b,
	__global float *c,
	const float k)
{
	size_t i = get_global_id(0);
	c[i] = (a[i] * k + b[i] * k) + k;
	printf("end");
}

__kernel void MatrixMul(
	int iHeightA,
	int iWidthA,
	__global float *pInMatA,
	int iHeightB,
	int iWidthB,
	__global float *pInMatB,
	__global float *pOutMat)
{
#if 0	
	int iRow = get_global_id(0);
	int iCol = get_global_id(1);
 
	float fSum = 0.0f;
 
	for (int i=0; i< iWidthA; i++)
	{
		fSum += pInMatA[iRow * iWidthA + i] * pInMatB[iWidthB * i + iCol];
	}
 
	pOutMat[iRow * iWidthB + iCol] = fSum;
#else
	int i = get_global_id(0);	
	int j = get_global_id(1); 	
	int k;	
	float tmp; 	
	if ((i < iHeightA) && (j < iWidthB)) // && (iWidthA == iHeightB)
	{		
		tmp = 0.0;
		for (k = 0; k < iWidthA; k++)
			tmp += pInMatA[i * iWidthA + k] * pInMatB[k * iWidthB + j];
		pOutMat[i * iHeightA + j] = tmp;
	}
#endif
}
