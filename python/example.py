#encoding: utf-8


import logging
import numpy as np
import os
import opencl4py as cl
import gc
import argparse
import datetime
import time

class testopencl(object):
    def readoclfile(filename):
        file_object = open(filename, 'r')
        oclfiledata = ""
        try:
            file_context = file_object.read()
            #oclfiledata = '\"\"\"\n' + file_context + '\"\"\"'
            oclfiledata = file_context
        finally:
            file_object.close()
        #print(oclfiledata)
        return oclfiledata
    def test1():
        logging.basicConfig(level=logging.DEBUG)
        platforms = cl.Platforms()
        print("OpenCL devices:\n\n%s\n"%platforms.dump_devices())
        ctx = platforms.create_some_context()
        queue = ctx.create_queue(ctx.devices[0])
        prg = ctx.create_program(
        """
        __kernel void test(__global const float *a, __global const float *b,
        __global float *c, const float k) {
        size_t i = get_global_id(0);
        c[i] = (a[i] + b[i]) * k;
        }
        """)
        krn = prg.get_kernel("test")
        a = np.arange(1000000, dtype=np.float32)
        b = np.arange(1000000, dtype=np.float32)
        c = np.empty(1000000, dtype=np.float32)
        k = np.array([0.5], dtype=np.float32)
        a_buf = ctx.create_buffer(cl.CL_MEM_READ_ONLY | cl.CL_MEM_COPY_HOST_PTR, a)
        b_buf = ctx.create_buffer(cl.CL_MEM_READ_ONLY | cl.CL_MEM_COPY_HOST_PTR, b)
        c_buf = ctx.create_buffer(cl.CL_MEM_WRITE_ONLY | cl.CL_MEM_ALLOC_HOST_PTR, size=c.nbytes)
        krn.set_arg(0, a_buf)
        krn.set_arg(1, b_buf)
        krn.set_arg(2, c_buf)
        krn.set_arg(3, k[0:1])
        queue.execute_kernel(krn, [a.size], None)
        queue.read_buffer(c_buf, c)
        diff = np.fabs(c - (a + b) * k[0])
        print(diff)
                     
    def test():
        print(os.environ.get("PYOPENCL_CTX"))
        os.environ["PYOPENCL_CTX"] = "0:0"      
        # Create platform, context, program, kernel and queue
        platforms = cl.Platforms()
        print("OpenCL devices:\n%s"%platforms.dump_devices())
        ctx = platforms.create_some_context()
        queue = ctx.create_queue(ctx.devices[0], cl.CL_QUEUE_PROFILING_ENABLE)  
        '''
        prg = ctx.create_program(
        """
        __kernel void test(
                __global float *a, 
                __global float *b, 
                const float c) 
        {
          size_t i = get_global_id(0);
          a[i] = (a[i] + b[i]) * c;
        }
        """)
        '''
        prg = ctx.create_program(testopencl.readoclfile("test.cl"))
        krn = prg.get_kernel("test")
              
        # Create arrays with some values for testing
        a = np.arange(100000, dtype=np.float32)
        b = np.cos(a)
        a = np.sin(a)
        a_copy = a.copy()
        # Prepare arrays for use with map_buffer
        a = cl.realign_array(a, queue.device.memalign, np)
        b = cl.realign_array(b, queue.device.memalign, np)
        c = np.array([0.1], dtype=np.float32)
        d = (a + b) * c[0]
        # Create buffers
        a_ = ctx.create_buffer(cl.CL_MEM_READ_WRITE | cl.CL_MEM_USE_HOST_PTR, a)
        b_ = ctx.create_buffer(cl.CL_MEM_READ_WRITE | cl.CL_MEM_USE_HOST_PTR, b)
        # Set kernel arguments
        krn.set_args(a_, b_, c[0:1])
        # Execute kernel
        global_size = [a.size]
        local_size = None
        queue.execute_kernel(krn, global_size, local_size, need_event=False)
         # Get results back from the device by map_buffer
        ev, ptr = queue.map_buffer(a_, cl.CL_MAP_READ, a.nbytes)
        del ev
        queue.unmap_buffer(a_, ptr).wait()
        print(a - d)
        aa = np.zeros(a.shape, dtype=a.dtype)
        queue.read_buffer(a_, aa)
        print(aa - d)
        # Refill buffer with stored copy by write_buffer
        ev = queue.write_buffer(a_, a_copy, blocking=False, need_event=True)
        # Execute kernel
        ev = queue.execute_kernel(krn, global_size, local_size, wait_for=(ev,))
        # Get results back from the device by map_buffer
        ev, ptr = queue.map_buffer(a_, cl.CL_MAP_READ, a.nbytes, wait_for=(ev,), need_event=True)
        ev.wait()
        queue.unmap_buffer(a_, ptr).wait()
        print(a - d)
        bb = np.zeros(a.shape, dtype=a.dtype)
        queue.read_buffer(a_, bb)
        print(bb - d)      
        del queue
        del ctx
        del krn
        del prg
        gc.collect()
        
    def testadd():        
        os.environ["PYOPENCL_CTX"] = "0:0"
        platforms = cl.Platforms()
        print("OpenCL devices:\n%s"%platforms.dump_devices())
        #ctx = platforms.create_some_context()
        ctx = cl.Context(platforms.platforms[0],
                         platforms.platforms[0].devices[0:1])        
        dev = ctx.devices[0]
        print("version=%s,\ngroup_size=%s"%(dev.version, dev.max_work_group_size))
        #prg = ctx.create_program(src_test, include_dirs)
        prg = ctx.create_program(testopencl.readoclfile("test.cl"))
        bins = prg.binaries[0]
        print(prg.kernel_names)
        #print(bins)
        krn = prg.get_kernel("testadd")
        print(krn.attributes)
        queue = ctx.create_queue(ctx.devices[0])        
        a = np.arange(100, dtype=np.float32)
        b = np.arange(100, dtype=np.float32)
        c = np.empty(100, dtype=np.float32)
        k = np.array([0.5], dtype=np.float32)
        a_buf = ctx.create_buffer(cl.CL_MEM_READ_ONLY | cl.CL_MEM_COPY_HOST_PTR, a)
        b_buf = ctx.create_buffer(cl.CL_MEM_READ_ONLY | cl.CL_MEM_COPY_HOST_PTR, b)
        c_buf = ctx.create_buffer(cl.CL_MEM_WRITE_ONLY | cl.CL_MEM_ALLOC_HOST_PTR, size=c.nbytes)        
        '''
        krn.set_arg(0, a_buf)
        krn.set_arg(1, b_buf)
        krn.set_arg(2, c_buf)
        krn.set_arg(3, k[0:1])
        '''
        krn.set_args(a_buf, b_buf, c_buf, k[0:1])
        ev = queue.execute_kernel(krn, [a.size], None)
        queue.read_buffer(c_buf, c)
        diff = c - (a * k[0]+ b * k[0]) * k[0]
        #print(a)
        #print(c)
        print(diff)   
        del queue
        del ctx
        del krn
        del prg
        gc.collect()
        
    def testmul():        
        os.environ["PYOPENCL_CTX"] = "0:0"
        platforms = cl.Platforms()
        print("OpenCL devices:\n%s"%platforms.dump_devices())
        ctx = platforms.create_some_context()
        prg = ctx.create_program(testopencl.readoclfile("test.cl"))
        print(prg.kernel_names)
        krn = prg.get_kernel("matmul")
        print(krn.attributes)
        queue = ctx.create_queue(ctx.devices[0])        
        a = np.arange(10, dtype=np.float32)
        b = np.arange(10, dtype=np.float32)
        c = np.empty(1000, dtype=np.float32)
        m = np.array([10], dtype=np.float32)
        p = np.array([10], dtype=np.float32)
        n = np.array([10], dtype=np.float32)
        a_buf = ctx.create_buffer(cl.CL_MEM_READ_ONLY | cl.CL_MEM_COPY_HOST_PTR, a)
        b_buf = ctx.create_buffer(cl.CL_MEM_READ_ONLY | cl.CL_MEM_COPY_HOST_PTR, b)
        c_buf = ctx.create_buffer(cl.CL_MEM_WRITE_ONLY | cl.CL_MEM_ALLOC_HOST_PTR, size=c.nbytes)        
        '''
        krn.set_arg(0, a_buf)
        krn.set_arg(1, b_buf)
        krn.set_arg(2, c_buf)
        krn.set_arg(3, m[0:1])
        krn.set_arg(4, p[0:1])
        krn.set_arg(5, n[0:1])
        '''
        krn.set_args(a_buf, b_buf, c_buf, m[0:1], p[0:1], c[0:1])
        #queue.execute_kernel(krn, [a.size], None)
        #queue.read_buffer(c_buf, c)
        #diff = np.fabs(c - (a * k[0]+ b * k[0]) * k[0])
        print(a)
        print(b)
        #print(diff)
        del queue
        del ctx
        del krn
        del prg
        gc.collect()
        
    def matrixmul():        
        os.environ["PYOPENCL_CTX"] = "0:0"
        platforms = cl.Platforms()
        print("OpenCL devices:\n%s"%platforms.dump_devices())
        ctx = platforms.create_some_context()
        prg = ctx.create_program(testopencl.readoclfile("test.cl"))
        print(prg.kernel_names)
        krn = prg.get_kernel("MatrixMul")
        print(krn.attributes)
        queue = ctx.create_queue(ctx.devices[0])
        
        iHeightA = np.array([800], dtype=np.int32)
        iWidthA = np.array([500], dtype=np.int32)
        pInMatA = np.arange(iHeightA[0] * iWidthA[0], dtype=np.float32)

        iHeightB = np.array([500], dtype=np.int32)
        iWidthB = np.array([800], dtype=np.int32)
        pInMatB = np.arange(iHeightB[0] * iWidthB[0], dtype=np.float32)

        pOutMat = np.empty(iHeightA[0] * iWidthB[0], dtype=np.float32)

        
        pInMatA_buf = ctx.create_buffer(cl.CL_MEM_READ_ONLY | cl.CL_MEM_COPY_HOST_PTR, pInMatA)
        pInMatB_buf = ctx.create_buffer(cl.CL_MEM_READ_ONLY | cl.CL_MEM_COPY_HOST_PTR, pInMatB)
        pOutMat_buf = ctx.create_buffer(cl.CL_MEM_READ_WRITE | cl.CL_MEM_ALLOC_HOST_PTR, size=pOutMat.nbytes)        

        krn.set_args(iHeightA[0:1], iWidthA[0:1], pInMatA_buf, iHeightB[0:1], iWidthB[0:1], pInMatB_buf, pOutMat_buf)
        global_size = [pInMatA.size, pInMatB.size]
        local_size = None
        for i in range(10):
            start = time.time()
            ev = queue.execute_kernel(krn, global_size, local_size, need_event=True)           
            t1 = time.time() - start   
            #ev, ptr = queue.map_buffer(pOutMat_buf, cl.CL_MAP_READ, pOutMat.nbytes)
            
            #queue.unmap_buffer(pOutMat_buf, ptr).wait()
            queue.read_buffer(pOutMat_buf, pOutMat)
                    
            data1 = np.reshape(pOutMat,(iHeightA[0] , iWidthB[0]))
            print(data1[0][1:5])
            start = time.time()
            data2 = np.dot(np.reshape(pInMatA, (iHeightA[0],iWidthA[0])),np.reshape(pInMatB, (iHeightB[0],iWidthB[0])))
            t2 = time.time() - start
            pInMatA += pInMatA
            ev = queue.write_buffer(pInMatA_buf, pInMatA, blocking=False, need_event=True)
            ev = queue.write_buffer(pInMatB_buf, pInMatB, blocking=False, need_event=True)            
            print(data2[0][1:5])
            print(t1,t2)
        del queue
        del ctx
        del krn
        del prg
        gc.collect()
        
if __name__ == "__main__":
    try:
        testopencl.matrixmul()
    finally:
        print('Opencl test end!!')
