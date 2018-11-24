#ifndef BENCHMARK_H
#define BENCHMARK_H

#include "../fio/load_tsv.h"
#include "../dformat/convert.h"
#include "../tools/Utils.h"
#include "../tools/Time.h"

#define FORMAT_NUM 4
#define CSR 0
#define BSR 1
#define HYB 2
#define ELL 3

#define ITER 10

static std::string format_names[] ={
		"CSR",
		"BSR",
		"HYB",
		"ELL"
};

static int formats[] = {
		CSR,
		BSR,
		HYB,
		ELL
};

#define UNIFIED_MEMORY 1
#define DEBUG_SPMV true

template<class Z, class T>
__global__ void set_one(T *values, Z nnz)
{
	Z tid = blockIdx.x * blockDim.x + threadIdx.x;

	if ( tid < nnz )
	{
		values [tid] = 1.0f;
	}
}

template<class Z>
class sort_indices_asc
{
   private:
     Z *rows;
     Z *cols;

   public:
     sort_indices_asc(Z* rows, Z* cols){
    	 this->rows = rows;
    	 this->cols = cols;
     }
     bool operator()(Z i, Z j) const {
    	 if(rows[i] == rows[j])
    	 {
    		 return cols[i]<cols[j];
    	 }else{
    		 return rows[i]<rows[j];
    	 }
     }
};

template<class Z, class T>
class BenchSPMV{
	public:
		BenchSPMV(int format){
			coo.row_idx = NULL; coo.col_idx = NULL; coo.values = NULL;
			csr.row_idx = NULL; csr.col_idx = NULL; csr.values = NULL;
		};

		~BenchSPMV(){
			free_coo();
			free_csr();
			free_bsr();
			free_hyb();
			clear_timers();
		};

		void load_coo(std::string fname);
		void coo_to_csr();
		void csr_to_bsr();
		void csr_to_hyb();

		void power_method();
	private:
		coo_format<Z,T> coo;
		csr_format<Z,T> csr;
		bsr_format<Z,T> bsr;

		uint64_t count_lines(std::string fname);
		void sort_coo();
		void sort_coo_host();

		void free_coo();
		void free_csr();
		void free_bsr();
		void free_hyb();

		void benchmark();
		void clear_timers();
		double tt_norm_val;
		double tt_norm;
		double tt_spmv;
		double tt_lambda;
		Time<millis> tt;

};

template<class Z, class T>
void BenchSPMV<Z,T>::benchmark()
{
	double tt_pmethod = this->tt_norm_val + this->tt_norm + this->tt_spmv + this->tt_lambda;
	std::cout << std::fixed << std::setprecision(4);
	std::cout << "<<<<<<<<<< -----------BENCHMARK----------- >>>>>>>>>>" << std::endl;
	std::cout << "Op(msec)\t:\tTotal(msec)\t:\tIter<x " << ITER << "> (msec)" << std::endl;
	std::cout << "norm_val  \t:\t"  << std::setfill('0') << std::setw(7+5) << this->tt_norm_val << "\t:\t" << std::setfill('0') << std::setw(7+5) << this->tt_norm_val/ITER << std::endl;
	std::cout << "norm     \t:\t" << std::setfill('0') << std::setw(7+5) << this->tt_norm << "\t:\t" << std::setfill('0') << std::setw(7+5) << this->tt_norm/ITER << std::endl;
	std::cout << "spmv     \t:\t" << std::setfill('0') << std::setw(7+5) << this->tt_spmv << "\t:\t" << std::setfill('0') << std::setw(7+5) << this->tt_spmv/ITER << std::endl;
	std::cout << "tt_lambda \t:\t" << std::setfill('0') << std::setw(7+5) << this->tt_lambda << "\t:\t" << std::setfill('0') << std::setw(7+5) << this->tt_lambda/ITER << std::endl;
	std::cout << "power method \t:\t" << std::setfill('0') << std::setw(7+5) << tt_pmethod << "\t:\t" << std::setfill('0') << std::setw(7+5) << tt_pmethod/ITER << std::endl;
	std::cout << "<<<<<<<<<< ------------------------------- >>>>>>>>>>" << std::endl;
}

template<class Z, class T>
void BenchSPMV<Z,T>::clear_timers()
{
	this->tt_norm_val = 0;
	this->tt_norm = 0;
	this->tt_spmv = 0;
	this->tt_lambda = 0;
}

template<class Z, class T>
void BenchSPMV<Z,T>::free_coo()
{
#if UNIFIED_MEMORY
	if(coo.row_idx != NULL){ cudaFreeHost(coo.row_idx); coo.row_idx = NULL; }
	if(coo.col_idx != NULL){ cudaFreeHost(coo.col_idx); coo.col_idx = NULL; }
	if(coo.values != NULL){ cudaFreeHost(coo.values); coo.values = NULL; }
#else
	if(coo.row_idx != NULL){ cudaFree(coo.row_idx); coo.row_idx = NULL; }
	if(coo.col_idx != NULL){ cudaFree(coo.col_idx); coo.col_idx = NULL; }
	if(coo.values != NULL){ cudaFree(coo.values); coo.values = NULL; }
#endif
}

template<class Z, class T>
void BenchSPMV<Z,T>::free_csr()
{
#if UNIFIED_MEMORY
	if(csr.row_idx != NULL) cudaFreeHost(csr.row_idx);
	if(csr.col_idx != NULL) cudaFreeHost(csr.col_idx);
	if(csr.values != NULL) cudaFreeHost(csr.values);
#else
	if(csr.row_idx != NULL) cudaFree(csr.row_idx);
	if(csr.col_idx != NULL) cudaFree(csr.col_idx);
	if(csr.values != NULL) cudaFree(csr.values);
#endif
}

template<class Z, class T>
void BenchSPMV<Z,T>::free_bsr()
{
#if UNIFIED_MEMORY
	if(bsr.bsrRowPtrA != NULL) cudaFreeHost(bsr.bsrRowPtrA);
	if(bsr.bsrColIndA != NULL) cudaFreeHost(bsr.bsrColIndA);
	if(bsr.bsrValA != NULL) cudaFreeHost(bsr.bsrValA);
#else
	if(bsr.bsrRowPtrA != NULL) cudaFree(bsr.bsrRowPtrA);
	if(bsr.bsrColIndA != NULL) cudaFree(bsr.bsrColIndA);
	if(bsr.bsrValA != NULL) cudaFree(bsr.bsrValA);
#endif
}

template<class Z, class T>
void BenchSPMV<Z,T>::free_hyb()
{
	if(hybA != NULL) cusparseDestroyHybMat(hybA);
}

template<class Z, class T>
uint64_t BenchSPMV<Z,T>::count_lines(std::string fname)
{
	uint64_t count = 0;
	FILE *fp = fopen(fname.c_str(), "r");
	long bytes_total = 0;
	long bytes_read = 0;
	long bytes_progress = BYTES_FRAME;
	if(fp == NULL){ perror(("error opening file " + fname).c_str()); exit(1); }
	fseek(fp,0,SEEK_END);
	bytes_total = ftell(fp);
	std::cout << "total(bytes): " << bytes_total << std::endl;
	rewind(fp);

	uint64_t size = 1024;
	char *buffer = (char*)malloc(sizeof(char)*size);
	int bytes = 0;

	do{//Count N
		for(int i = 0; i < bytes; i++) if( buffer[i] == '\n') count++;
		bytes_read = ftell(fp);
		uint32_t p = bytes_read / bytes_progress;
		if( p > 0 )
		{
			//std::cout << "Loading: [" << (uint32_t)((((double)bytes_read)/(bytes_total))*100) << "] -- " << edges.size() <<" \r";
			std::cout << "Loading: [" << (uint32_t)((((double)bytes_read)/(bytes_total))*100) << "] \r";
			std::cout.flush();
			bytes_progress += BYTES_FRAME;
		}
	}while( (bytes = fread(buffer,sizeof(char),size, fp)) > 0 );

	fclose(fp);
	free(buffer);

	return count;
}

template<class Z, class T>
void BenchSPMV<Z,T>::sort_coo_host()
{
	Z *indices = (Z*)malloc(sizeof(Z)*coo.nnz);
	for(uint64_t i = 0; i < coo.nnz; i++) indices[i];
	std::cout << "Sorting coo indices ..." << std::endl;
	std::sort(indices, indices+coo.nnz, sort_indices(coo.row_idx,coo.col_idx));
}

template<class Z, class T>
void BenchSPMV<Z,T>::sort_coo()
{
	Z *d_P = NULL;
	void *pBuffer = NULL;
	size_t pBufferSizeInBytes = 0;
	cusparse_status = cusparseXcoosort_bufferSizeExt(
			cusparse_handle,
			coo.m,
			coo.m,
			coo.nnz,
			coo.row_idx,
			coo.col_idx,
			&pBufferSizeInBytes
			);
	cusp_util::handle_error(cusparse_status,"coosort buffersizeext");
	printf("pBufferSizeInBytes = %lld bytes \n", (long long)pBufferSizeInBytes);
	cutil::safeMallocHost<void,uint64_t>(&(pBuffer),sizeof(char)*pBufferSizeInBytes,"buffer alloc");
	cutil::safeMallocHost<Z,uint64_t>(&(d_P),sizeof(Z)*coo.nnz,"d_P");

	//Initialize permutation vector
	cusparse_status = cusparseCreateIdentityPermutation(
			cusparse_handle,
			coo.nnz,
			d_P
			);
	cusp_util::handle_error(cusparse_status,"coo cusparseCreateIdentityPermutation");

	//sort COO format by Row
	cusparse_status = cusparseXcoosortByRow(
				cusparse_handle,
				coo.m,
				coo.m,
				coo.nnz,
				coo.row_idx,
				coo.col_idx,
				d_P,
				pBuffer
			);
	cusp_util::handle_error(cusparse_status,"coo cusparseXcoosortByRow");
	//Deallocate memory
	cudaFreeHost(pBuffer);
	cudaFreeHost(d_P);
}

template<class Z, class T>
void BenchSPMV<Z,T>::load_coo(std::string fname)
{
	std::cout << "Counting edges ... " << std::endl;
	uint64_t edge_num = count_lines(fname);
	//uint64_t edge_num =1468365182;
	coo.nnz = edge_num;

	std::cout << "Edges ... " << edge_num << std::endl;
	#if UNIFIED_MEMORY
	cutil::safeMallocHost<Z,uint64_t>(&(coo.row_idx),sizeof(Z)*coo.nnz,"coo coo.row_idx alloc");//ROW IDX FOR COO
	cutil::safeMallocHost<Z,uint64_t>(&(coo.col_idx),sizeof(Z)*coo.nnz,"coo coo.col_idx alloc");//ROW IDX FOR COO
	#else
	coo.row_idx = (Z*)malloc(sizeof(Z)*coo.nnz);//Allocate coo.row_idx
	coo.col_idx = (Z*)malloc(sizeof(Z)*coo.nnz);//Allocate coo.col_idx
	#endif
	//////////////////////////////////////////////////////
	//Load//
	FILE *fp = fopen(fname.c_str(), "r");
	long bytes_total = 0;
	long bytes_read = 0;
	long bytes_progress = BYTES_FRAME;
	if(fp == NULL){ perror(("error opening file " + fname).c_str()); exit(1); }
	fseek(fp,0,SEEK_END);
	bytes_total = ftell(fp);
	std::cout << "total(bytes): " << bytes_total << std::endl;
	rewind(fp);

	uint32_t n[2] = {0,0};
	uint32_t mx[2] = {0,0};
	uint32_t mn[2] = {UINT_MAX,UINT_MAX};
	std::cout << "Loading edges from " << fname << " !!!" <<std::endl;
	std::cout << "Loading: [" << (uint32_t)((((double)bytes_read)/(bytes_total))*100) << "]\r";
	std::cout.flush();
	uint64_t i = 0;
	bool not_sorted = false;
	while(fscanf(fp,"%u\t%u",&n[0],&n[1]) > 0)
	{
		//std::cout << n[0] << "-->" << n[1] << std::endl;
		mx[0] = std::max(n[0],mx[0]); mx[1] = std::max(n[1],mx[1]);
		mn[0] = std::min(n[0],mn[0]); mn[1] = std::min(n[1],mn[1]);
		coo.row_idx[i] = n[0];
		coo.col_idx[i] = n[1];
		i++;

		bytes_read = ftell(fp);
		uint32_t p = bytes_read / bytes_progress;
		if( p > 0 )
		{
			//std::cout << "Loading: [" << (uint32_t)((((double)bytes_read)/(bytes_total))*100) << "] -- " << edges.size() <<" \r";
			std::cout << "Loading: [" << (uint32_t)((((double)bytes_read)/(bytes_total))*100) << "] \r";
			std::cout.flush();
			bytes_progress += BYTES_FRAME;
		}
		if(i > 1 && coo.row_idx[i-1] > coo.row_idx[i])
		{
			not_sorted = true;
		}
	};
	std::cout.flush();
	if(DEBUG_SPMV) std::cout << "max: " << mx[0] << "," << mx[1] << std::endl;
	if(DEBUG_SPMV) std::cout << "min: " << mn[0] << "," << mn[1] << std::endl;
	idx_bounds.mx = std::max(mx[0],mx[1]);
	idx_bounds.mn = std::min(mn[0],mn[1]);
	fclose(fp);

	//Sort coo//
	coo.m = (idx_bounds.mx - idx_bounds.mn + 1);
	if(not_sorted){
		if(DEBUG_SPMV) do{ std::cout << '\n' << "Acknowledge sorting coo indices (large memory)..."; } while (std::cin.get() != '\n');
		sort_coo();
	}
	//////////////////////////////////////////////////////
	if(DEBUG_SPMV){
		std::cout << "Making index base 0 ... " << std::endl;
		for(uint64_t i = 0; i < coo.nnz; i++)
		{
			if (i < 350){ std::cout << coo.row_idx[i] << " -- " << coo.col_idx[i] << std::endl; }
			coo.row_idx[i] -= idx_bounds.mn;
			coo.col_idx[i] -= idx_bounds.mn;
		}
	}
}

template<class Z, class T>
void BenchSPMV<Z,T>::coo_to_csr()
{
	//COO Data
	std::cout << "Converting coo to csr ... " << coo.m << std::endl;
#if !UNIFIED_MEMORY
	Z *coo_row_idx = NULL;
	cutil::safeMalloc<Z,uint64_t>(&(coo_row_idx),sizeof(Z)*coo.nnz,"coo row_idx alloc");//ROW IDX FOR COO
	cutil::safeCopyToDevice<Z,uint64_t>(coo_row_idx, coo.row_idx,sizeof(Z)*coo.nnz, "coo.row_idx copy to coo_row_idx");
#endif

	//CSR Data
	csr.m = coo.m;
	csr.nnz = coo.nnz;
#if !UNIFIED_MEMORY
	cutil::safeMalloc<Z,uint64_t>(&(csr.row_idx),sizeof(Z)*(csr.m+1),"csr row_idx alloc");//ROW IDX FOR CSR
#else
	cutil::safeMallocHost<Z,uint64_t>(&(csr.row_idx),sizeof(Z)*(csr.m+1),"csr row_idx alloc");//ROW IDX FOR CSR
#endif

	std::cout << "Calculating row indices ..." <<std::endl;
#if !UNIFIED_MEMORY
	cusparse_status = cusparseXcoo2csr(cusparse_handle, coo_row_idx, coo.nnz, coo.m, csr.row_idx, CUSPARSE_INDEX_BASE_ZERO);
#else
	cusparse_status = cusparseXcoo2csr(cusparse_handle, coo.row_idx, coo.nnz, coo.m, csr.row_idx, CUSPARSE_INDEX_BASE_ZERO);
#endif
	cusp_util::handle_error(cusparse_status,"csr row indices");

	std::cout << "Initializing csr col indices ..." << std::endl;
#if !UNIFIED_MEMORY
	cudaFree(coo_row_idx);
	cutil::safeMalloc<Z,uint64_t>(&(csr.col_idx),sizeof(Z)*coo.nnz,"csr col indices alloc");
	cutil::safeCopyToDevice<Z,uint64_t>(csr.col_idx, coo.col_idx,sizeof(Z)*coo.nnz, "csr copy to csr.col_idx");
#else
	cutil::safeMallocHost<Z,uint64_t>(&(csr.col_idx),sizeof(Z)*coo.nnz,"csr col indices alloc");
	memcpy(csr.col_idx, coo.col_idx,sizeof(Z)*coo.nnz);
	//csr.col_idx = coo.col_idx;
#endif
	std::cout << std::endl;
	free_coo();
}

template<class Z, class T>
void BenchSPMV<Z,T>::csr_to_bsr()
{
	//coo_to_csr();//TODO:comment out since csr is first to be created//

	bsr.blockDim = 4;
	bsr.dir = CUSPARSE_DIRECTION_COLUMN;
	bsr.m = csr.m;
	bsr.mb = (bsr.m + bsr.blockDim - 1)/bsr.blockDim;
#if !UNIFIED_MEMORY
	cutil::safeMalloc<Z,uint64_t>(&(bsr.bsrRowPtrA),sizeof(Z)*(bsr.mb+1),"bsrRowPtrC alloc");
#else
	cutil::safeMallocHost<Z,uint64_t>(&(bsr.bsrRowPtrA),sizeof(Z)*(bsr.mb+1),"bsrRowPtrC alloc");
#endif
	//Calculate nnzb
	int *nnzTotalDevHostPtr = &bsr.nnzb;
	cusparse_status = cusparseXcsr2bsrNnz(cusparse_handle, bsr.dir, bsr.m, bsr.m,
				cusparse_descrA,
				csr.row_idx, csr.col_idx,
				bsr.blockDim,
				cusparse_descrA,
				bsr.bsrRowPtrA,
				nnzTotalDevHostPtr
			);
	cusp_util::handle_error(cusparse_status,"bsrNnz error");

	if (nnzTotalDevHostPtr != NULL){
		bsr.nnzb = *nnzTotalDevHostPtr;
	}else{
	#if !UNIFIED_MEMORY
		cutil::safeCopyToHost<int,uint64_t>(&bsr.nnzb, bsr.bsrRowPtrA+bsr.mb,sizeof(int), "copy nnzb to host");
		cutil::safeCopyToHost<int,uint64_t>(&bsr.base, bsr.bsrRowPtrA,sizeof(int), "copy base to host");
	#else
		bsr.nnzb = bsr.bsrRowPtrA[bsr.mb];
		bsr.base = bsr.bsrRowPtrA[0];
	#endif
		bsr.nnzb -= bsr.base;
	}

	//Allocate space for columns and values
	#if !UNIFIED_MEMORY
		cutil::safeMalloc<Z,uint64_t>(&(bsr.bsrColIndA),sizeof(Z)*bsr.nnzb,"bsrColIndC alloc");
		cutil::safeMalloc<T,uint64_t>(&(bsr.bsrValA),sizeof(T)*bsr.nnzb,"bsrValC alloc");
	#else
		cutil::safeMallocHost<Z,uint64_t>(&(bsr.bsrColIndA),sizeof(Z)*bsr.nnzb,"bsrColIndC alloc");
		cutil::safeMallocHost<T,uint64_t>(&(bsr.bsrValA),sizeof(T)*bsr.nnzb,"bsrValC alloc");
	#endif

	std::cout << "Converting data to bsr ..." << std::endl;
	cusparse_status = cusparseDcsr2bsr(cusparse_handle, bsr.dir, bsr.m, bsr.m,
				cusparse_descrA,
				csr.values, csr.row_idx, csr.col_idx,
				bsr.blockDim,
				cusparse_descrA,
				bsr.bsrValA, bsr.bsrRowPtrA, bsr.bsrColIndA
			);
	cusp_util::handle_error(cusparse_status,"csr to bsr");
}

template<class Z, class T>
void BenchSPMV<Z,T>::csr_to_hyb()
{
	//Create hyb mat
	cusparse_status = cusparseCreateHybMat(&hybA);
	cusp_util::handle_error(cusparse_status,"create hyb mat");

	//csr to hyb conversion//
	cusparse_status = cusparseDcsr2hyb(cusparse_handle, csr.m, csr.m, cusparse_descrA,
				csr.values, csr.row_idx, csr.col_idx,
	            hybA,
				0, //required when partitionType == CUSPARSE_HYB_PARTITION_USER
	            CUSPARSE_HYB_PARTITION_AUTO);//partitionType: CUSPARSE_HYB_PARTITION_AUTO, CUSPARSE_HYB_PARTITION_USER, CUSPARSE_HYB_PARTITION_MAX

	cusp_util::handle_error(cusparse_status,"csr to hyb");
}

template<class Z, class T>
void BenchSPMV<Z,T>::power_method()
{
	T *dx, *dy;
	T nrm2_x;
	T h_one = 1.0f;
	T h_zero = 0.0f;
	T lambda_next = 0.0f;

	//Allocation and Random Initialization//
	coo_to_csr();
#ifndef UNIFIED_MEMORY
 	cutil::safeMalloc<T,uint64_t>(&(dx),sizeof(T)*coo.m,"dx values alloc");
	cutil::safeMalloc<T,uint64_t>(&(dy),sizeof(T)*coo.m,"dy values alloc");
	cutil::safeMalloc<T,uint64_t>(&(csr.values),sizeof(T)*csr.nnz,"coo.values alloc");//TODO:Initialize to one
	dim3 csr_init_block(256,1,1);
	dim3 csr_init_grid((csr.nnz - 1)/256 + 1,1,1);
	set_one<<<csr_init_grid,csr_init_block>>>(csr.values,csr.nnz);
	cutil::cudaCheckErr(cudaDeviceSynchronize(),"Error executing set_one");
#else
 	cutil::safeMallocHost<T,uint64_t>(&(dx),sizeof(T)*coo.m,"dx values alloc");
	cutil::safeMallocHost<T,uint64_t>(&(dy),sizeof(T)*coo.m,"dy values alloc");
	cutil::safeMallocHost<T,uint64_t>(&(csr.values),sizeof(T)*coo.nnz,"coo.values alloc");//TODO:
	//for(uint32_t i = 0; i < csr.nnz; i++) csr.values[i] = 1.0f;
#endif

	cutil::cudaInitRandStates();
	cutil::cudaRandInit<T,uint64_t>(dx,csr.m);
	cusparse_status = cusparseCreateMatDescr(&cusparse_descrA);
	cusp_util::handle_error(cusparse_status,"create matrix descrA");
    cusparseSetMatIndexBase(cusparse_descrA,CUSPARSE_INDEX_BASE_ZERO);
    cusparseSetMatType(cusparse_descrA, CUSPARSE_MATRIX_TYPE_GENERAL );

	for(uint32_t i = 0; i < 3; i++)
	{
		int format = formats[i];

		switch(format){
			case CSR:
				std::cout << "<<<<< Testing CSR Format >>>>>" << std::endl;
				break;
			case BSR:
				std::cout << "<<<<< Testing BSR Format >>>>>" << std::endl;
				csr_to_bsr();
				break;
			case HYB:
				std::cout << "<<<<< Testing HYB Format >>>>>" << std::endl;
				csr_to_hyb();
				break;
			default:
				std::cout << "FORMAT <" << format_names[format] << "> NOT SUPPORTED!!!" << std::endl;
				break;
		}

		//euclidean norm of dx
		for(int j = 0; j < ITER; j++){
			tt.start();
			if(DEBUG_SPMV) std::cout << "calculate normalize value ..." << std::endl;
			cublas_status = cublasDnrm2_v2(cublas_handle, csr.m, dx, 1, &nrm2_x );
			cublas_util::handle_error(cublas_status,"calculate normalize value");
			this->tt_norm_val += tt.lap();

			//normalize dx
			tt.start();
			if(DEBUG_SPMV) std::cout << "normalize vector ..." << std::endl;
			T one_over_nrm2_x = 1.0 / nrm2_x;
			cublas_status = cublasDscal_v2(cublas_handle, csr.m, &one_over_nrm2_x, dx, 1 );
			cublas_util::handle_error(cublas_status,"normalize vector");
			this->tt_norm += tt.lap();

			if(DEBUG_SPMV) std::cout << "y = A*x" << std::endl;
			this->tt.start();
			switch(format){
				case CSR:
					cusparse_status = cusparseDcsrmv_mp(cusparse_handle,
                                         	 CUSPARSE_OPERATION_NON_TRANSPOSE,
                                         	 csr.m,//rows
										 	 csr.m,//cols
                                         	 csr.nnz,//nnz
                                         	 &h_one,//scalar
										 	 cusparse_descrA,
                                         	 csr.values,
                                         	 csr.row_idx,
                                         	 csr.col_idx,
                                         	 dx,
                                         	 &h_zero,
                                         	 dy);
					cusp_util::handle_error(cusparse_status,"csr_spmv execution");
					break;
				case BSR:
					cusparse_status = cusparseDbsrmv(
											cusparse_handle,
											bsr.dir,
											CUSPARSE_OPERATION_NON_TRANSPOSE,
											bsr.mb, bsr.mb, bsr.nnzb,
											&h_one,
											cusparse_descrA,
											bsr.bsrValA, bsr.bsrRowPtrA, bsr.bsrColIndA, bsr.blockDim, dx, &h_zero, dy);
					break;
				case HYB:
					cusparse_status = cusparseDhybmv(
											cusparse_handle,
											CUSPARSE_OPERATION_NON_TRANSPOSE,
											&h_one,
											cusparse_descrA,
											hybA, dx, &h_zero, dy);

					break;
				default:
					std::cout << "FORMAT <" << format_names[format] << "> NOT SUPPORTED!!!" << std::endl;
			}
			this->tt_spmv += tt.lap();

			this->tt.start();
			if(DEBUG_SPMV) std::cout << "lambda = y**T*x" << std::endl;
			cublas_status = cublasDdot_v2 (cublas_handle, csr.m, dx, 1, dy, 1, &lambda_next);
			cublas_util::handle_error(cublas_status,"calculate lambda = y**T*x");
			this->tt_lambda += tt.lap();
		}
		this->benchmark();
		this->clear_timers();
	}

#ifndef UNIFIED_MEMORY
	cudaFree(dx);
	cudaFree(dy);
#else
	cudaFreeHost(dx);
	cudaFreeHost(dy);
#endif
	cusparse_status = cusparseDestroyMatDescr(cusparse_descrA);
	cusp_util::handle_error(cusparse_status,"destroy matrix descrA");

	return ;

//	//Initialize vector
//	T *x = (T*)malloc(sizeof(T)*csr.m);
//	for(uint64_t i = 0; i < csr.m; i++) x[i] = 0.5;
//	cutil::safeCopyToDevice<T,uint64_t>(dx, x,sizeof(T)*csr.m, "x copy to dx");
//	free(x);
//
//	//////////////////////////
//	//Set Matrix Descriptors//
//	cusparse_status = cusparseCreateMatDescr(&cusparse_descrA);
//	cusp_util::handle_error(cusparse_status,"create matrix descrA");
//    cusparseSetMatIndexBase(cusparse_descrA,CUSPARSE_INDEX_BASE_ZERO);
//    cusparseSetMatType(cusparse_descrA, CUSPARSE_MATRIX_TYPE_GENERAL );
//
//    std::cout << "Executing Pagerank ..." <<std::endl;
//	if( std::is_same<T,double>::value )
//	{
//		//euclidean norm of dx
//		std::cout << "calculate normalize value ..." << std::endl;
//		cublas_status = cublasDnrm2_v2(cublas_handle, csr.m, dx, 1, &nrm2_x );
//		cublas_util::handle_error(cublas_status,"calculate normalize value");
//
//		//normalize dx
//		std::cout << "normalize vector ..." << std::endl;
//		T one_over_nrm2_x = 1.0 / nrm2_x;
//		cublas_status = cublasDscal_v2(cublas_handle, csr.m, &one_over_nrm2_x, dx, 1 );
//		cublas_util::handle_error(cublas_status,"normalize vector");
//
//		std::cout << "y = A*x" << std::endl;
//		cusparse_status = cusparseDcsrmv_mp(cusparse_handle,
//                                         CUSPARSE_OPERATION_NON_TRANSPOSE,
//                                         csr.m,
//										 csr.m,
//                                         csr.nnz,
//                                         &h_one,
//										 cusparse_descrA,
//                                         csr.values,
//                                         csr.row_idx,
//                                         csr.col_idx,
//                                         dx,
//                                         &h_zero,
//                                         dy);
//		cusp_util::handle_error(cusparse_status,"csr_spmv execution");
//
//		std::cout << "lambda = y**T*x" << std::endl;
//		cublas_status = cublasDdot_v2 (cublas_handle, csr.m, dx, 1, dy, 1, &lambda_next);
//		cublas_util::handle_error(cublas_status,"calculate lambda = y**T*x");

//	}
//

//	destroy_csr<Z,T>(csr);
}

#endif
