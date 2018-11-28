#NVCC=/usr/local/cuda-8.0/bin/nvcc
NVCC=nvcc
NVCC_LIBS=-L/usr/local/cuda-9.2/lib64/
#NVCC_LIBS=-L/usr/local/cuda-9.0/targets/x86_64-linux/lib/

GEXE=gmain

#NVCC_FLAGS = --ptxas-options=-v -gencode arch=compute_35,code=sm_35 -rdc=true
#ARCH = -gencode arch=compute_61,code=sm_61
#ARCH = -gencode arch=compute_35,code=sm_35
CUDA_LIBS = -lcusparse_static -lculibos -lcublas_static

all:
	$(NVCC) $(NVCC_LIBS) -std=c++11 $(CUDA_LIBS) main.cu -o $(GEXE) $(NVCC_INCLUDE)
		
ptx:
	$(NVCC) -std=c++11 $(ARCH) $(CUDA_LIBS) -ptx main.cu
	#./ptx_chain
	
dryrun:
	$(NVCC) -dryrun -std=c++11 $(CUBLAS_LIB) $(ARCH) gpu_sgemm/main.cu -o $(GEXE) --keep 2>dryrunout
	
clean:
	rm -rf $(GEXE)
	rm -rf main.ptx
	rm -rf gmain_*
	rm -rf main.cpp*
	rm -rf *.ii
	rm -rf *.cubin
	rm -rf *.gpu
	rm -rf *.cudafe*
	rm -rf *.o
	rm -rf *fatbin*
	rm -rf *module_id
	rm -rf mm_main
	
