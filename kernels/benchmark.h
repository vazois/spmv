#ifndef BENCHMARK_H
#define BENCHMARK_H

#include "../fio/load_tsv.h"
#include "../dformat/convert.h"
#include "../tools/Utils.h"

#define FORMAT_NUM 4
#define CSR 0
#define ELL 1
#define HYB 2
#define BSR 3

static std::string format_names[] ={
		"CSR",
		"ELL",
		"HYB",
		"BSR"
};

static int formats[] = {
		CSR,
		ELL,
		HYB,
		BSR
};

#define UNIFIED_MEMORY 0


template<class Z, class T>
class BenchSPMV{
	public:
		BenchSPMV(int format){
			coo.row_idx = NULL; coo.col_idx = NULL; coo.values = NULL;
			csr.row_idx = NULL; csr.col_idx = NULL; csr.values = NULL;
		};

		~BenchSPMV(){
		#if UNIFIED_MEMORY
			if(coo.row_idx != NULL) cudaFreeHost(coo.row_idx); if(coo.col_idx != NULL) cudaFreeHost(coo.col_idx); if(coo.values != NULL) cudaFreeHost(coo.values);
			if(csr.row_idx != NULL) cudaFreeHost(csr.row_idx); if(csr.col_idx != NULL) cudaFreeHost(csr.col_idx); if(csr.values != NULL) cudaFreeHost(csr.values);
		#else
			if(coo.row_idx != NULL) free(coo.row_idx); if(coo.col_idx != NULL) free(coo.col_idx); if(coo.values != NULL) free(coo.values);
			if(csr.row_idx != NULL) cudaFree(csr.row_idx); if(csr.col_idx != NULL) cudaFree(csr.col_idx); if(csr.values != NULL) cudaFree(csr.values);
		#endif

		};

		void load_coo(std::string fname);
		void coo_to_csr_old();
		void coo_to_csr();

		void power_method();

	private:
		coo_format<Z,T> coo;
		csr_format<Z,T> csr;

		uint64_t count_lines(std::string fname);
};

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
void BenchSPMV<Z,T>::load_coo(std::string fname)
{
	std::cout << "Counting edges ... " << std::endl;
	uint64_t edge_num = count_lines(fname);
	coo.nnz = edge_num;

	std::cout << "Edges ... " << edge_num << std::endl;
#if UNIFIED_MEMORY
	cutil::safeMallocHost<Z,uint64_t>(&(coo.row_idx),sizeof(Z)*coo.nnz,"coo coo.row_idx alloc");//ROW IDX FOR COO
	cutil::safeMallocHost<Z,uint64_t>(&(coo.col_idx),sizeof(Z)*coo.nnz,"coo coo.col_idx alloc");//ROW IDX FOR COO
	cutil::safeMallocHost<T,uint64_t>(&(coo.values),sizeof(T)*coo.nnz,"coo coo.values alloc");//ROW IDX FOR COO
#else
	coo.row_idx = (Z*)malloc(sizeof(Z)*coo.nnz);//Allocate coo.row_idx
	coo.col_idx = (Z*)malloc(sizeof(Z)*coo.nnz);//Allocate coo.col_idx
	coo.values = (T*)malloc(sizeof(T)*coo.nnz);//Allocate coo.values
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
	};
	std::cout.flush();
	std::cout << "max: " << mx[0] << "," << mx[1] << std::endl;
	std::cout << "min: " << mn[0] << "," << mn[1] << std::endl;
	idx_bounds.mx = std::max(mx[0],mx[1]);
	idx_bounds.mn = std::min(mn[0],mn[1]);
	fclose(fp);

	//////////////////////////////////////////////////////
	std::cout << "Making index base 0 ... " << std::endl;
	for(uint64_t i = 0; i < coo.nnz; i++)
	{
		coo.row_idx[i] -= idx_bounds.mn;
		coo.col_idx[i] -= idx_bounds.mn;
		coo.values[i] = 1.0f;
	}
	coo.m = (idx_bounds.mx - idx_bounds.mn + 1);
}

template<class Z, class T>
void BenchSPMV<Z,T>::coo_to_csr_old()
{
	//COO Data
	std::cout << "Converting ... " << coo.m << std::endl;
	Z *coo_row_idx = NULL;
	cutil::safeMalloc<Z,uint64_t>(&(coo_row_idx),sizeof(Z)*coo.nnz,"coo row_idx alloc");//ROW IDX FOR COO
	cutil::safeCopyToDevice<Z,uint64_t>(coo_row_idx, coo.row_idx,sizeof(Z)*coo.nnz, "coo.row_idx copy to coo_row_idx");

	//CSR Data
	csr.m = coo.m;
	csr.nnz = coo.nnz;
	cutil::safeMalloc<Z,uint64_t>(&(csr.row_idx),sizeof(Z)*(csr.m+1),"csr row_idx alloc");//ROW IDX FOR CSR

	std::cout << "Converting data to csr ..." <<std::endl;
	cusparse_status = cusparseXcoo2csr(cusparse_handle, coo_row_idx, coo.nnz, coo.m, csr.row_idx, CUSPARSE_INDEX_BASE_ZERO);
	cusp_util::handle_error(cusparse_status,"coo to csr");

//	pp<Z,T><<<1,1>>>(csr.row_idx,csr.m);
//	cudaDeviceSynchronize();
	cudaFree(coo_row_idx);

	cutil::safeMalloc<Z,uint64_t>(&(csr.col_idx),sizeof(Z)*coo.nnz,"csr col_idx alloc");//COL IDX FOR CSR
	cutil::safeMalloc<T,uint64_t>(&(csr.values),sizeof(T)*coo.nnz,"csr values alloc");
	cutil::safeCopyToDevice<Z,uint64_t>(csr.col_idx, coo.col_idx,sizeof(Z)*coo.nnz, "csr copy to csr.col_idx");
	cutil::safeCopyToDevice<T,uint64_t>(csr.values, coo.values,sizeof(T)*coo.nnz, "csr copy to csr.values");
}

template<class Z, class T>
void BenchSPMV<Z,T>::coo_to_csr()
{
	//COO Data
	std::cout << "Converting ... " << coo.m << std::endl;
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

	std::cout << "Converting data to csr ..." <<std::endl;
#if !UNIFIED_MEMORY
	cusparse_status = cusparseXcoo2csr(cusparse_handle, coo_row_idx, coo.nnz, coo.m, csr.row_idx, CUSPARSE_INDEX_BASE_ZERO);
#else
	cusparse_status = cusparseXcoo2csr(cusparse_handle, coo.row_idx, coo.nnz, coo.m, csr.row_idx, CUSPARSE_INDEX_BASE_ZERO);
#endif
	cusp_util::handle_error(cusparse_status,"coo to csr");

//	pp<Z,T><<<1,1>>>(csr.row_idx,csr.m);
//	cudaDeviceSynchronize();
#if !UNIFIED_MEMORY
	cudaFree(coo_row_idx);
	cutil::safeMalloc<Z,uint64_t>(&(csr.col_idx),sizeof(Z)*coo.nnz,"csr col_idx alloc");//COL IDX FOR CSR
	cutil::safeMalloc<T,uint64_t>(&(csr.values),sizeof(T)*coo.nnz,"csr values alloc");
	cutil::safeCopyToDevice<Z,uint64_t>(csr.col_idx, coo.col_idx,sizeof(Z)*coo.nnz, "csr copy to csr.col_idx");
	cutil::safeCopyToDevice<T,uint64_t>(csr.values, coo.values,sizeof(T)*coo.nnz, "csr copy to csr.values");
#else
	csr.col_idx = coo.col_idx;
	csr.values = coo.values;
#endif

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
#ifndef UNIFIED_MEMORY
 	cutil::safeMalloc<T,uint64_t>(&(dx),sizeof(T)*coo.m,"dx values alloc");
	cutil::safeMalloc<T,uint64_t>(&(dy),sizeof(T)*coo.m,"dy values alloc");
#else
 	cutil::safeMallocHost<T,uint64_t>(&(dx),sizeof(T)*coo.m,"dx values alloc");
	cutil::safeMallocHost<T,uint64_t>(&(dy),sizeof(T)*coo.m,"dy values alloc");
#endif

	cutil::cudaInitRandStates();
	cutil::cudaRandInit<T,uint64_t>(dx,coo.m);
	cusparse_status = cusparseCreateMatDescr(&cusparse_descrA);
	cusp_util::handle_error(cusparse_status,"create matrix descrA");
    cusparseSetMatIndexBase(cusparse_descrA,CUSPARSE_INDEX_BASE_ZERO);
    cusparseSetMatType(cusparse_descrA, CUSPARSE_MATRIX_TYPE_GENERAL );

	for(uint32_t i = 0; i < 1; i++)
	{
		int format = formats[i];

		switch(format){
			case CSR:
				coo_to_csr();
				break;
			default:
				std::cout << "FORMAT <" << format_names[format] << "> NOT SUPPORTED!!!" << std::endl;
		}
		//return ;
		//euclidean norm of dx
		std::cout << "calculate normalize value ..." << std::endl;
		cublas_status = cublasDnrm2_v2(cublas_handle, csr.m, dx, 1, &nrm2_x );
		cublas_util::handle_error(cublas_status,"calculate normalize value");

		//normalize dx
		std::cout << "normalize vector ..." << std::endl;
		T one_over_nrm2_x = 1.0 / nrm2_x;
		cublas_status = cublasDscal_v2(cublas_handle, csr.m, &one_over_nrm2_x, dx, 1 );
		cublas_util::handle_error(cublas_status,"normalize vector");

		switch(format){
			case CSR:
				std::cout << "y = A*x" << std::endl;
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
			default:
				std::cout << "FORMAT <" << format_names[format] << "> NOT SUPPORTED!!!" << std::endl;
		}

		std::cout << "lambda = y**T*x" << std::endl;
		cublas_status = cublasDdot_v2 (cublas_handle, csr.m, dx, 1, dy, 1, &lambda_next);
		cublas_util::handle_error(cublas_status,"calculate lambda = y**T*x");
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
