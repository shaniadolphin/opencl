# 使用opencl
标签： python opencl

****
|Author|shaniadolphin|
|:---:|:---|
|E-mail|349948204@qq.com|
****

## 矩阵乘法
&emsp;&emsp;对于以下是一个常见的线性方程组，
![pi](http://latex.codecogs.com/png.latex?\left\{\n\begin{array}{c}\na_{11}x_{1}+b_{12}x_{2} =y_1\\\na_{21}x_{1}+b_{22}x_{2} =y_2\\\n\end{array}\n\right.)

$$
\left \{ 
\begin{array}{c}
a_{11}x_{1}+b_{12}x_{2} =y_1 \\ 
a_{21}x_{1}+b_{22}x_{2} =y_2 \\ 
\end{array}
\right.
$$
&emsp;&emsp;用矩阵表示就是：
$$
\begin{pmatrix}a_{11} & b_{12} \\ b_{a1} & b_{22}\\ \end{pmatrix} \begin{pmatrix}x_{1} \\ x_{2}\\ \end{pmatrix}=\begin{pmatrix}y_{1} \\ y_{2}\\ \end{pmatrix}
$$
&emsp;&emsp;推导出矩阵乘法的计算：
$$
\begin{pmatrix}a_{11} & a_{12} \\ a_{21} & a_{22}\\ \end{pmatrix} \begin{pmatrix}b_{11} & b_{12} \\ b_{21} & b_{22}\\ \end{pmatrix}=
\begin{pmatrix}a_{11}b_{11}+a_{21}b_{12} & a_{11}b_{12}+a_{12}b_{22} \\ a_{21}a_{11}+a_{22}a_{21} & a_{21}b_{12}+a_{22}b_{22} \end{pmatrix}
$$
&emsp;&emsp;进一步的，使用一般的表现形式（A为m\*n，B为n\*p）：
$$
(AB)_{ij}=\sum_{r=1}^na_{ir}b_{rj}=a_{i1}b_{1j}+a_{i2}b_{2j}+\cdots+a_{in}b_{nj}
$$
&emsp;&emsp;表示了矩阵A和矩阵B相乘后，新矩阵各元素等于该元素所在A的列与B的行对应相乘后进行累加。
## OpenCL的程序
&emsp;&emsp;OpenCL是一个为异构平台编写程序的框架，支持大量不同的应用，提供了两种层面的并行机制：任务并行与数据并行，用来加快数据的处理。
&emsp;&emsp;以下是OpenCL编程中常见的词条，规范了编程所涉及对象的名称和概念：
* &emsp;&emsp;Platform (平台)：主机加上OpenCL框架管理下的若干设备，通过平台，主机应用程序可以与设备共享资源并在设备上执行kernel。基本上一个厂商对应一个Platform，比如Intel, AMD，Mali，PowerVR，一个主机可以有多个平台，比如有独显的笔记本同时有核显和独显。
* &emsp;&emsp;Device（设备）：计算单元（Compute Units）的集合，GPU是典型的device。但是Intel和AMD的多核CPU也提供OpenCL接口，所以也可以作为Device，所以一个平台上可以有多个设备，比如PC上有两个英伟达的显卡。
* &emsp;&emsp;Context（上下文）：OpenCL的Platform上共享和使用资源的环境，包括kernel、device、memory objects、command queue等。一般一个Platform对应一个Context。
* &emsp;&emsp;Command Queue（指令队列）：在指定设备上管理多个指令（Command）。队列里指令执行可以顺序也可以乱序。一个设备可以对应多个指令队列。
* &emsp;&emsp;Program：OpenCL程序，由kernel函数、其他函数、声明和变量等组成。
* &emsp;&emsp;Kernel（内核函数）：运行在设备端的函数。
* &emsp;&emsp;Memory Object（内存对象）：在主机和设备之间传递数据的对象，一般映射到OpenCL程序中的global memory。有两种具体的类型：Buffer Object（缓存对象）和Image Object（图像对象）。
* &emsp;&emsp;NDRange：主机端运行设备端kernel函数的主要接口。
* &emsp;&emsp;WaitForEvents（同步）：在将一个 OpenCL 命令提交到命令队列的时候，用来标识是以阻塞还是非阻塞的方式执行，如果后续处理依赖该内核函数返回的数据就要使用阻塞的方式，否则线程可以马上处理其它的事情，以提高并行效率。 

&emsp;&emsp;面向异构平台的应用一般按以下步骤实现：
* &emsp;&emsp;查找构成异构系统的平台（clGetPlatformIDs）；
* &emsp;&emsp;获得平台特征，使kernel函数能够适应不同硬件单元的特定特性以获得最优性能（clGetDeviceIDs）；
* &emsp;&emsp;创建上下文，建立平台设备在主机端的调用接口（clCreateContext）；
* &emsp;&emsp;创建将在平台上运行的程序（clCreateProgramWithSource、clBuildProgram、clCreateKernel）；
* &emsp;&emsp;创建指令队列（clCreateCommandQueue）；
* &emsp;&emsp;建立和管理涉及的内存对象（clCreateBuffer、clEnqueueWriteBuffer）；
* &emsp;&emsp;传递kernel函数的参数（clSetKernelArg）；
* &emsp;&emsp;执行kernel函数（clEnqueueNDRangeKernel）；
* &emsp;&emsp;主机同步（clWaitForEvents、clReleaseEvent）；
* &emsp;&emsp;获取执行结果（clEnqueueReadBuffer）；
* &emsp;&emsp;释放平台资源（clReleaseProgram、clReleaseContext、clReleaseCommandQueue、clReleaseDevice、clReleaseKernel）。

&emsp;&emsp;下图表示了系统中CPU和GPU的互联关系，包括用CPU为GPU创建程序，传递数据，而GPU则执行程序，运算数据，前文介绍了如何在CPU端构建应用，那要如何编程实现运行在GPU的内核函数呢？
![opencl框架](https://upload-images.jianshu.io/upload_images/11747660-4e515b6328195416.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/600)
&emsp;&emsp;首先，来理解一下OpenCL里的工作组、工作项和处理单元的概念：
* &emsp;&emsp;工作项：一个循环中最里面的一次运算，称为一个工作项。
* &emsp;&emsp;工作组：是由访问相同处理资源的工作项组成，包括工作项访问高速内存（也叫局部内存）的同一块内存和同步。
* &emsp;&emsp;处理单元：支持工作组的处理资源被称为处理单元，各个工作组在单个处理单元上的执行，各个处理单元一次只能够执行一个工作组。
&emsp;&emsp;处理单元的数量由硬件决定，OpenCL的内核函数应根据处理单元数量合理设置工作组和工作项，而工作项往往是一个具体的运算，比如矩阵乘法里的乘加运算。
&emsp;&emsp;使用openCL实现矩阵乘法运算的逻辑如下：
&emsp;&emsp;在OpenCL程序中，我们为每个工作项分配一个要计算的乘法矩阵的元素。将i，j的外层循环替换为函数调用工作项ID（get_global_id），并保证得到的工作项ID在矩阵C的范围内计算每个a[i][r]和每个b[r][j]的乘积，再对所有的乘积项求和，代码如下，其中多了参数有效性的判断。
```C
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
```
&emsp;&emsp;下面用C语言来实现这个算法，对比可知，最外层循环的i和j即上文中传入的工作项ID，它们表示这个在C语言里顺序执行的运算被内核函数并行处理了，比如在这个示例里两个相乘矩阵的第一个矩阵的Height和第二个矩阵的Width（两个矩阵相乘要求第一个矩阵的Width等于第二个矩阵的Height）共同构成了并行的。
```C
#define M 800
#define P 500
#define N 800

void RunAsCpu(const float* pInMatA,const float* pInMatB,float* pOutMat){
	for (int i = 0; i < M; i++){
		for (int j = 0; j < N; j++){
			pOutMat[i*N + j] = 0.0;
			for (int k = 0; k < P; k++){
				pOutMat[i*N + j] += pInMatA[i*P + k] * pInMatB[k*N + j];
			}
		}
	}
}
```
&emsp;&emsp;矩阵乘法的核心是一个乘加计算（`pOutMat[i*N + j] += pInMatA[i*P + k] * pInMatB[k*N + j]` ）。所以优化矩阵运算需要尽量减少数据移动，以保证这个计算可以接近峰值性能运行，参考[文章](https://blog.csdn.net/cloud_desktop/article/details/19822025)中的过程和方法。

## C语言实现
&emsp;&emsp;下载[Intel SDK for OpenCL applications](https://software.intel.com/en-us/intel-opencl/download)，选择平台后注册帐号后即可下载。
![Intel SDK下载](https://upload-images.jianshu.io/upload_images/11747660-10709f08885daca4.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/600)

&emsp;&emsp;从github的[intel_OpenCL_MulMatrix](https://github.com/shaniadolphin/opencl/tree/master/vs/intel_OpenCL_MulMatrix)下载工程，用VS打开[CapsBasic](https://github.com/shaniadolphin/opencl/tree/master/vs/intel_OpenCL_MulMatrix/CapsBasic "CapsBasic")中的sln工程即编译运行，下文内容中的代码引用自该工程。
&emsp;&emsp;以下代码用于找到用于运算的指定的GPU平台，本文所用的是Intel平台GPU。
```C
	cl_uint num_of_platforms = 0;
	error = clGetPlatformIDs(0, 0, &num_of_platforms);
	cl_platform_id* platforms = new cl_platform_id[num_of_platforms];
	error = clGetPlatformIDs(num_of_platforms, platforms, 0);
	cl_uint selected_platform_index = num_of_platforms;
	for (cl_uint i = 0; i < num_of_platforms; ++i){
		size_t platform_name_length = 0;
		size_t maxComputeUnits = 0;
		error = clGetDeviceInfo(deviceIds[0],CL_DEVICE_MAX_COMPUTE_UNITS,sizeof(cl_uint),&maxComputeUnits,NULL);
		error = clGetPlatformInfo(platforms[i],CL_PLATFORM_NAME,0,0,&platform_name_length);
		char* platform_name = new char[platform_name_length];
		error = clGetPlatformInfo(platforms[i],CL_PLATFORM_NAME,platform_name_length,platform_name,0);
		if (strstr(platform_name, "Intel") &&selected_platform_index == num_of_platforms)
			selected_platform_index = i;
		delete[] platform_name;
	}
	if (selected_platform_index == num_of_platforms) return 1;
```
&emsp;&emsp;然后就可以构建用于运算的queue，由平台platform获得context，再由clCreateCommandQueue将其构成queue：
```C
	cl_platform_id platform = platforms[selected_platform_index];
	error = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
	context = clCreateContext(0, 1, &device, NULL, NULL, &error);
	queue = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &error);
```
&emsp;&emsp;生成用于测试的数据，实际工作中，可以为图像数据：
```C
	float* A_h = new float[M*P];
	float* B_h = new float[P*N];
	float* C_h = new float[M*N];
	srand(100);
	for (int i = 0; i < M*P; i++)A_h[i] = (float)(rand() % 50);
	for (int i = 0; i < P*N; i++)B_h[i] = (float)(rand() % 50);
```
&emsp;&emsp;将生成的数据传入显存中，使用了clCreateBuffer函数，可以理解为将大小为sizeof(float)*M*P的A_h数组传入context的A_d显存中：
```C
		cl_mem A_d = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float)*M*P, A_h, &error);
		cl_mem B_d = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float)*P*N, B_h, &error);
		cl_mem C_d = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float)*M*N, NULL, &error);
```
&emsp;&emsp;载入openCL的源程序，通过fopen打开文件，用fread将文件中的内容读到source中：
```C
	FILE* fp = fopen("OpenCLMulMatrix.cl", "rb");
	fseek(fp, 0, SEEK_END);
	size_t src_size = ftell(fp);
	fseek(fp, 0, SEEK_SET);
	const char* source = new char[src_size];
	fread((void*)source, 1, src_size, fp);
	fclose(fp);
```
&emsp;&emsp;编译openCL的源程序，使用 clCreateProgramWithSource函数将程序所在的source传给context，再通过clBuildProgram函数进行编译，最终从编译的GPU可执行文件中找到指定的功能函数，如“RunAsGpu_1”，指定给cl_kernel所定义的run_as_gpu_1进行数据运算：
```C
	cl_program program = clCreateProgramWithSource(context, 1, &source, &src_size, &error);
	delete[] source;
	error = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
	char* build_log;
	size_t log_size;
	clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
	build_log = new char[log_size + 1];
	clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, build_log, NULL);
	build_log[log_size] = '\0';
	printf("build_log:%s", build_log);
	delete[] build_log;
	cl_kernel run_as_gpu_1 = clCreateKernel(program, "RunAsGpu_1", &error);
```
&emsp;&emsp;传入kernel程序运行的参数，通过clSetKernelArg，按*.cl文件中RunAsGpu_1函数所对应的参数顺序进行设定：
```C
	cl_int M_d = M;
	cl_int P_d = P;
	cl_int N_d = N;
	error = clSetKernelArg(run_as_gpu_1, 0, sizeof(cl_mem), &A_d);
	error |= clSetKernelArg(run_as_gpu_1, 1, sizeof(cl_mem), &B_d);
	error |= clSetKernelArg(run_as_gpu_1, 2, sizeof(int), &M_d);
	error |= clSetKernelArg(run_as_gpu_1, 3, sizeof(int), &N_d);
	error |= clSetKernelArg(run_as_gpu_1, 4, sizeof(int), &P_d);
	error |= clSetKernelArg(run_as_gpu_1, 5, sizeof(cl_mem), &C_d);
```
&emsp;&emsp;设定工作组和工作项，clEnqueueNDRangeKernel将在设备上对数据进行划分，运行kernel程序，对数据进行运算：
```C
	size_t globalws_1[2] = { M,N };
	cl_event ev;
	error = clEnqueueNDRangeKernel(queue, run_as_gpu_1, 2, NULL, globalws_1, NULL, 0, NULL, &ev);
	clFinish(queue);
```
```C
clEnqueueNDRangeKernel(cl_command_queue queue, cl_kernel kernel, 
			cl_uint work_dims, const size_t *global_work_offset, 
			const size_t *global_work_size, const size_t *local_work_size,
			cl_uint num_events, const cl_event *wait_list,cl_event *event)
```
* `work_dims` ：数据的维数，其限定范围一般是1-3；
* `global_work_offset` ：各个维度上全局ID偏移量，即数据的起始点，比如你希望二维数据的起始点是(1,3)，就可以通过这个参数设置；
* `global_work_size` ：工作项的总体数量。
* `local_work_size` ：一个工作组中工作项的数量，如果参数local_work_size的取值被设置成NULL，opencl将分析决定如何在设备上的处理单元间分配工作项。
&emsp;&emsp;比如，对于一个12x12的矩阵，将其划分为9个工作组，每个工作组是一个4x4的矩阵，即16个工作项，相应的设置：
```C
	cl_uint work_dims =2; //2维数据
	size_t global_work_offset[2] = {0,0}; //从(0,0)开始
	size_t global_work_size[2] = {12,12};
	size_t local_work_size[2] = {4,4};
	clEnqueueNDRangeKernel(command_q,kernel,work_dim,global_work_offset,global_work_size,local_work_size,0,NULL,NULL);
```
&emsp;&emsp;kernel运行的起始时间由kernel运行时自动生成，通过clGetEventProfilingInfo读取：
```C
	cl_ulong startTime, endTime;
	clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_START,sizeof(cl_ulong), &startTime, NULL);
	clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_END,sizeof(cl_ulong), &endTime, NULL);
	cl_ulong kernelExecTimeNs = endTime - startTime;
```
&emsp;&emsp;从GPU内存中取得运算结果，从queue中将显存gpu_C_1中的内容读到内存C_d中：
```C
	float* gpu_C_1 = new float[M*N];
	clEnqueueReadBuffer(queue, C_d, CL_TRUE, 0, M*N*sizeof(float), gpu_C_1, 0, NULL, NULL);
```
&emsp;&emsp;释放资源：
```C
	delete[] A_h;
	delete[] B_h;
	delete[] C_h;
	delete[] gpu_C_1;
	delete[] platforms;
	clReleaseKernel(run_as_gpu_1);
	clReleaseCommandQueue(queue);
	clReleaseContext(context);
	clReleaseMemObject(A_d);
	clReleaseMemObject(B_d);
	clReleaseMemObject(C_d);
```
&emsp;&emsp;运行时可以看到资源管理器中GPU的资源占用有明显上升。
![GPU资源查看](https://upload-images.jianshu.io/upload_images/11747660-c4f0e96d05590cbe.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/600)


## python实现
&emsp;&emsp;python中的操作与C可以完全对应。需要安装用于PC的opencl4py模块，在github上有这个模块的[源码](https://github.com/Samsung/opencl4py)：
```Bash
        pip install opencl4py
```
![opencl4py](https://upload-images.jianshu.io/upload_images/11747660-9050fbb1ab536a3d.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/600)

&emsp;&emsp;找到用于运算的指定的GPU平台：
```Python
        os.environ["PYOPENCL_CTX"] = "0:0"
        platforms = cl.Platforms()
        print("OpenCL devices:\n%s"%platforms.dump_devices())
```
&emsp;&emsp;然后就可以构建用于运算的queue，由平台platform获得context，再由clCreateCommandQueue将其构成queue：
```Python
        ctx = platforms.create_some_context()
        prg = ctx.create_program(testopencl.readoclfile("test.cl"))
        print(prg.kernel_names)
        krn = prg.get_kernel("MatrixMul")
        print(krn.attributes)
        queue = ctx.create_queue(ctx.devices[0])
```
&emsp;&emsp;生成用于测试的数据，实际工作中，可以为图像数据：
```Python
        iHeightA = np.array([800], dtype=np.int32)
        iWidthA = np.array([500], dtype=np.int32)
        pInMatA = np.arange(iHeightA[0] * iWidthA[0], dtype=np.float32)
        iHeightB = np.array([500], dtype=np.int32)
        iWidthB = np.array([800], dtype=np.int32)
        pInMatB = np.arange(iHeightB[0] * iWidthB[0], dtype=np.float32)
        pOutMat = np.empty(iHeightA[0] * iWidthB[0], dtype=np.float32)
```
&emsp;&emsp;将生成的数据传入显存中，使用了clCreateBuffer函数，可以理解为将大小为sizeof(float)\*M\*P的A_h数组传入context的A_d显存中：
```Python   
        pInMatA_buf = ctx.create_buffer(cl.CL_MEM_READ_ONLY | cl.CL_MEM_COPY_HOST_PTR, pInMatA)
        pInMatB_buf = ctx.create_buffer(cl.CL_MEM_READ_ONLY | cl.CL_MEM_COPY_HOST_PTR, pInMatB)
        pOutMat_buf = ctx.create_buffer(cl.CL_MEM_READ_WRITE | cl.CL_MEM_ALLOC_HOST_PTR, size=pOutMat.nbytes)        
```
&emsp;&emsp;创建一个readoclfile函数，载入openCL的源程序：
```Python
    def readoclfile(filename):
        file_object = open(filename, 'r')
        oclfiledata = ""
        try:
            file_context = file_object.read()
            oclfiledata = file_context
        finally:
            file_object.close()
        return oclfiledata
```
&emsp;&emsp;通过调用创建运行于内核的程序：
```Python
        prg = ctx.create_program(testopencl.readoclfile("test.cl"))
        print(prg.kernel_names)
```
指定用于内核程序中的具体功能函数：
```Python
        krn = prg.get_kernel("MatrixMul")
        print(krn.attributes)
```
&emsp;&emsp;传入kernel程序运行的参数，通过 krn.set_args按*.cl文件中功能函数所对应的参数顺序进行设定：
```Python
        krn.set_args(iHeightA[0:1], iWidthA[0:1], pInMatA_buf, iHeightB[0:1], iWidthB[0:1], pInMatB_buf, pOutMat_buf)
```
&emsp;&emsp;也可以像C程序一样逐个设定：
```Python
        krn.set_arg(0, iHeightA[0:1])
        krn.set_arg(1, iWidthA[0:1])
        krn.set_arg(2, pInMatA_buf)
        krn.set_arg(3, iHeightB[0:1])
        krn.set_arg(4, iWidthB[0:1])
        krn.set_arg(5, pInMatB_buf)
        krn.set_arg(5, pOutMat_buf)
```
&emsp;&emsp;运行kernel程序：
```Python
       start = time.time()
       ev = queue.execute_kernel(krn, global_size, local_size, need_event=True)           
       t1 = time.time() - start 
```
&emsp;&emsp;从GPU内存中取得运算结果：
```Python
       queue.read_buffer(pOutMat_buf, pOutMat)
       data1 = np.reshape(pOutMat,(iHeightA[0] , iWidthB[0]))
       print(data1[0][1:5])
```
&emsp;&emsp;打印由numpy运算的矩阵乘法数据，进行查看和比较：
```Python
       data2 = np.dot(np.reshape(pInMatA, (iHeightA[0],iWidthA[0])),np.reshape(pInMatB, (iHeightB[0],iWidthB[0])))
       print(data2[0][1:5]) 
```
&emsp;&emsp;释放资源：
```Python
        del queue
        del ctx
        del krn
        del prg
        gc.collect()
```
&emsp;&emsp;以下是程序的终端输出：
```Bash
OpenCL devices:
Platform 0: NVIDIA CUDA
	Device 0: GeForce GTX 970 (4096 Mb, 4096 align, OpenCL C 1.2)
['test', 'matmul', 'testadd', 'MatrixMul']

[3.3233502e+10 3.3233648e+10 3.3233791e+10 3.3233891e+10]
[3.3233508e+10 3.3233648e+10 3.3233781e+10 3.3233904e+10]
0.015639781951904297 0.0
[6.6467004e+10 6.6467295e+10 6.6467582e+10 6.6467783e+10]
[6.6467017e+10 6.6467295e+10 6.6467561e+10 6.6467807e+10]
0.0 0.0
```
&emsp;&emsp;输出的信息包括主机所用到的OPENCL平台和设备型号，比如上面为英伟达的GTX970，所编译的用于GPU的功能名称包括“test”，“matmul”，“testadd”和“matrixmul”。
&emsp;&emsp;运行时可以看到资源管理器中GPU的资源占用有明显上升。
![GPU资源查看](https://upload-images.jianshu.io/upload_images/11747660-92fc66c55ee08a45.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/600)


### **参考文档**

|#|链接地址|文档名称|
|:---:|---|---|
|1|`https://github.com/Samsung/opencl4py`|[opencl4py](https://github.com/Samsung/opencl4py "GITHUB")|
|2|`https://www.cnblogs.com/Reyzal/p/7389993.html`|[Intel核心显卡OpenCL环境搭建](https://www.cnblogs.com/Reyzal/p/7389993.html "CSDN文章")|
|3|`https://www.khronos.org/registry/OpenCL/sdk/1.2/docs/man/xhtml/`|[OpenCL参考手册](https://www.khronos.org/registry/OpenCL/sdk/1.2/docs/man/xhtml/ "官方文章")|
|4|`https://blog.csdn.net/cloud_desktop/article/details/19822025`|[优化OpenCL矩阵运算](https://blog.csdn.net/cloud_desktop/article/details/19822025 "CSDN文章")|
|5|`https://blog.csdn.net/LIYUAN123ZHOUHUI/article/details/52850282`|[opencl中工作组和工作项](https://blog.csdn.net/LIYUAN123ZHOUHUI/article/details/52850282 "CSDN文章")|

****

<script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=default"></script>
