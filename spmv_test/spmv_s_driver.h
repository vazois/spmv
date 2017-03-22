#ifndef SPMV_DRIVER_H
#define SPMV_DRIVER_H

#include "../cuda/CudaHelper.h"
#include "../cuda/CuSparseHelper.h"

#include "spmv_kernels.h"

void zeros(float *&X, uint64_t size){
	for(uint64_t i = 0;i<size;i++) X[i] = 0;
}

void cpu_coo(float *&hY, float *hX, int *hR, int *hC, float *hD, uint64_t nnz, uint64_t vsize){
	for(uint64_t i = 0; i< nnz;i++){
		uint64_t rIndex = hR[i];
		uint64_t cIndex = hC[i];
		hY[rIndex] += hD[i] * hX[cIndex];

	}
}

template<typename DATA_T>
void print_vector(DATA_T *hX, uint64_t limit, uint64_t vsize){
	std::cout<<"<START>\n";
	if(limit > vsize){ std::cout<<"(ERROR) vector access out of bound!!!" << std::endl; return;}
	for(uint64_t i = 0;i<limit;i++){
		std::cout<<hX[i] << std::endl;
	}
	std::cout<<"<END>\n";
}

void small_example(int *&hR,int *&hC, float *&hD, float *&hX, float *&hY, uint64_t vsize){
	hR[0] = 0; hR[1] = 0; hR[2] = 1; hR[3] = 1; hR[4] = 2; hR[5] = 2; hR[6] = 3;
	hC[0] = 1; hC[1] = 2; hC[2] = 0; hC[3] = 1; hC[4] = 0; hC[5] = 3; hC[6] = 2;
	hD[0] = 1; hD[1] = 2; hD[2] = 1; hD[3] = 3; hD[4] = 1; hD[5] = 4; hD[6] = 2;
	hX[0] = 1; hX[1] = 2; hX[2] = 3; hX[3] = 4;
	zeros(hY,vsize);
}

void small_test_coo(){
	cudaDeviceReset();
	cusparseHandle_t handle;
	cusparseMatDescr_t descr=0;
	int *hR,*hC; float *hD;
	int *dR,*dC; float *dD;

	float *hX,*hY;
	float *dX,*dY;
	int *csrRowPtr;
	int *csrRowPtrHost;

	uint64_t nnz = 7;
	uint64_t vsize = 4;
	uint64_t rnum = 4;

	cusp_util::handle_error(cusparseCreate(&handle),"handle create");
	cusp_util::handle_error(cusparseCreateMatDescr(&descr),"error creating mat descr");
	cusparseSetMatType(descr,CUSPARSE_MATRIX_TYPE_GENERAL);
	cusparseSetMatIndexBase(descr,CUSPARSE_INDEX_BASE_ZERO);

	cutil::safeMallocHost<int,uint64_t>(&hR,sizeof(int)*nnz,"hR memory alloc");//rows host
	cutil::safeMallocHost<int,uint64_t>(&hC,sizeof(int)*nnz,"hC memory alloc");//cols host
	cutil::safeMallocHost<float,uint64_t>(&hD,sizeof(float)*nnz,"hD memory alloc");//data host

	cutil::safeMallocHost<float,uint64_t>(&hX,sizeof(float)*vsize,"hX memory alloc");
	cutil::safeMallocHost<float,uint64_t>(&hY,sizeof(float)*vsize,"hY memory alloc");

	cutil::safeMalloc<int,uint64_t>(&dR,sizeof(int)*nnz,"dR memory alloc");//rows device
	cutil::safeMalloc<int,uint64_t>(&dC,sizeof(int)*nnz,"dC memory alloc");//cols device
	cutil::safeMalloc<float,uint64_t>(&dD,sizeof(float)*nnz,"dD memory alloc");//data device

	cutil::safeMalloc<float,uint64_t>(&dX,sizeof(float)*vsize,"dX memory alloc");
	cutil::safeMalloc<float,uint64_t>(&dY,sizeof(float)*vsize,"dY memory alloc");

	cutil::safeMalloc<int,uint64_t>(&csrRowPtr,sizeof(int)*(rnum+1),"csrRowPtr memory alloc");
	cutil::safeMallocHost<int,uint64_t>(&csrRowPtrHost,sizeof(int)*(rnum+1),"csrRowPtrHost memory alloc");
	//return;

	small_example(hR,hC,hD,hX,hY,vsize);

	cutil::safeCopyToDevice<int,uint64_t>(dR,hR,sizeof(int)*nnz, "Error copying hR to dR");
	cutil::safeCopyToDevice<int,uint64_t>(dC,hC,sizeof(int)*nnz, "Error copying hC to dC");
	cutil::safeCopyToDevice<float,uint64_t>(dD,hD,sizeof(float)*nnz, "Error copying hD to dD");
	cutil::safeCopyToDevice<float,uint64_t>(dX,hX,sizeof(float)*vsize, "Error copying hX to dX");
	cutil::safeCopyToDevice<float,uint64_t>(dY,hY,sizeof(float)*vsize, "Error copying hY to dY");

	cpu_coo(hY,hX,hR,hC,hD,nnz,vsize);
	print_vector<float>(hY,vsize,vsize);

	cusp_util::handle_error(cusparseXcoo2csr(handle,dR,nnz,rnum,csrRowPtr,CUSPARSE_INDEX_BASE_ZERO),"csr2coo conversion");
	cutil::cudaCheckErr(cudaDeviceSynchronize(),"conversion synchronize");
	cutil::safeCopyToHost<int,uint64_t>(csrRowPtrHost,csrRowPtr,sizeof(int)*(rnum+1), "Error copying csrRowPtr to csrRowPtrHost");
	print_vector<int>(csrRowPtrHost,vsize,vsize);

	float alpha = 1.0;
	float beta = 1.0;
	cusparseScsrmv(
			handle,CUSPARSE_OPERATION_NON_TRANSPOSE,
			rnum,rnum,nnz,&alpha,
			descr,
			dD,
			csrRowPtr,dC,
			dX,&beta,
			dY
	);

	cutil::safeCopyToHost<float,uint64_t>(hY,dY,sizeof(float)*vsize, "Error copying dY to hY");
	cutil::cudaCheckErr(cudaDeviceSynchronize(),"csrmv synchronize");
	print_vector<float>(hY,vsize,vsize);

	cudaFreeHost(hR); cudaFreeHost(hC); cudaFreeHost(hD);
	cudaFreeHost(hX); cudaFreeHost(hY);
	cudaFree(dR); cudaFree(dC); cudaFree(dD);
	cudaFree(dX); cudaFree(dY);
	cudaFreeHost(csrRowPtrHost);
	cudaFree(csrRowPtr);
	cusp_util::handle_error(cusparseDestroy(handle),"handle destroy");
}

#endif
