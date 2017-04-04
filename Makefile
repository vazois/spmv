CPUC=g++
#NVCC=/usr/local/cuda-8.0/bin/nvcc
NVCC=/usr/local/cuda-7.5/bin/nvcc

CEXE=cmain
GEXE=gmain

CBLAS_FLAGS = -lgsl -lcblas -l
COPT_FLAGS= -O3 -ffast-math -funroll-loops -msse -mmmx -fomit-frame-pointer -m64
INCLUDE_DIR=-I /home/vzois/git/openblas/ -L/home/vzois/git/openblas/ -lopenblas

#NVCC_FLAGS = --ptxas-options=-v -gencode arch=compute_35,code=sm_35 -rdc=true
#ARCH = -gencode arch=compute_61,code=sm_61
ARCH = -gencode arch=compute_35,code=sm_35
CUDA_LIBS = -lcusparse_static -lculibos

all:
	$(CPUC) fio/main.cpp fio/mmio.c -o mm_main
		
spmv_t:
	$(NVCC) -std=c++11 -O3 $(ARCH) $(CUDA_LIBS) spmv_test/main.cu -o $(GEXE)
	
ptx:
	$(NVCC) -std=c++11 $(ARCH) $(CUDA_LIBS) -ptx spmv_test/main.cu
	#./ptx_chain
	
dryrun:
	$(NVCC) -dryrun -std=c++11 $(CUBLAS_LIB) $(ARCH) gpu_sgemm/main.cu -o $(GEXE) --keep 2>dryrunout
	
clean:
	rm -rf $(CEXE)
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
	