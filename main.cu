#include "tools/ArgParser.h"

#include "fio/load_tsv.h"
#include "dformat/convert.h"

template<class Z, class T>
void coo_benchmark()
{
	coo_format<Z,T> coo;
	edges_to_coo<Z,T>(coo,edges);
	
	destroy_coo<Z,T>(coo);
}

template<class Z, class T>
void csr_benchmark()
{
	csr_format<Z,T> csr;
	edges_to_csr<Z,T>(csr,edges);
	
	destroy_csr<Z,T>(csr);
}

template<class Z, class T>
void power_method_csr()
{
	csr_format<Z,T> csr;
	edges_to_csr<Z,T>(csr,edges);
	T *dx, *dy;
	T nrm2_x;
	T h_one = 1.0f;
	T h_zero = 0.0f;
	T lambda_next = 0.0f;
	cutil::safeMalloc<T,uint64_t>(&(dx),sizeof(T)*csr.m,"dx values alloc");
	cutil::safeMalloc<T,uint64_t>(&(dy),sizeof(T)*csr.m,"dy values alloc");
	
	//Initialize vector
	T *x = (T*)malloc(sizeof(T)*csr.m);
	for(uint64_t i = 0; i < csr.m; i++) x[i] = 0.5;
	cutil::safeCopyToDevice<T,uint64_t>(dx, x,sizeof(T)*csr.m, "x copy to dx");
	free(x);
	
	//////////////////////////
	//Set Matrix Descriptors//
	cusparse_status = cusparseCreateMatDescr(&cusparse_descrA);
	cusp_util::handle_error(cusparse_status,"create matrix descrA");
    cusparseSetMatIndexBase(cusparse_descrA,CUSPARSE_INDEX_BASE_ZERO);
    cusparseSetMatType(cusparse_descrA, CUSPARSE_MATRIX_TYPE_GENERAL );
	
    std::cout << "Executing Pagerank ..." <<std::endl;
	if( std::is_same<T,double>::value )
	{
		//euclidean norm of dx
		std::cout << "calculate normalize value ..." << std::endl;
		cublas_status = cublasDnrm2_v2(cublas_handle, csr.m, dx, 1, &nrm2_x );
		cublas_util::handle_error(cublas_status,"calculate normalize value");
		
		//normalize dx
		std::cout << "normalize vector ..." << std::endl;
		T one_over_nrm2_x = 1.0 / nrm2_x;
		cublas_status = cublasDscal_v2(cublas_handle, csr.m, &one_over_nrm2_x, dx, 1 );
		cublas_util::handle_error(cublas_status,"normalize vector");
		
		std::cout << "y = A*x" << std::endl;
		cusparse_status = cusparseDcsrmv_mp(cusparse_handle,
                                         CUSPARSE_OPERATION_NON_TRANSPOSE,
                                         csr.m,
										 csr.m,
                                         csr.nnz,
                                         &h_one,
										 cusparse_descrA,
                                         csr.values,
                                         csr.row_idx,
                                         csr.col_idx,
                                         dx,
                                         &h_zero,
                                         dy);
		cusp_util::handle_error(cusparse_status,"csr_spmv execution");
		
		std::cout << "lambda = y**T*x" << std::endl;
		cublas_status = cublasDdot_v2 (cublas_handle, csr.m, dx, 1, dy, 1, &lambda_next);
		cublas_util::handle_error(cublas_status,"calculate lambda = y**T*x");
		
	}
	
	cudaFree(dx);
	cudaFree(dy);
	cusparse_status = cusparseDestroyMatDescr(cusparse_descrA);
	cusp_util::handle_error(cusparse_status,"destroy matrix descrA");
	destroy_csr<Z,T>(csr);
}

int main(int argc, char **argv){
	ArgParser ap;
	ap.parseArgs(argc,argv);
	
	if(!ap.exists("-f"))
	{
		std::cout << "Missing input file name. Specify with -f=<full_path>!!!" << std::endl;
		exit(1);
	}
	std::string fname = ap.getString("-f");
	
	//////////////
	//initialize//
	//////////////
	cusparse_status = cusparseCreate(&cusparse_handle);
	cusp_util::handle_error(cusparse_status,"create handle cusparse");
	
	cublas_status = cublasCreate_v2(&cublas_handle);
	cublas_util::handle_error(cublas_status,"create handle cublas");
	/////////////
	//Load File//
	/////////////
	load_tsv(fname);

	//coo_benchmark<uint64_t,double>();
	power_method_csr<int,double>();
	
	///////////
	//destroy//
	///////////
	cusparse_status = cusparseDestroy(cusparse_handle);
	cusp_util::handle_error(cusparse_status,"destroy handle cusparse");
	cublas_status = cublasDestroy(cublas_handle);
	cublas_util::handle_error(cublas_status,"destroy handle cublas");
	
	return 0;
}