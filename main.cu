#include "tools/ArgParser.h"
#include "kernels/benchmark.h"


int main(int argc, char **argv){
	ArgParser ap;
	ap.parseArgs(argc,argv);
	
	if(!ap.exists("-f"))
	{
		std::cout << "Missing input file name. Specify with -f=<full_path>!!!" << std::endl;
		exit(1);
	}
	std::string fname = ap.getString("-f");
	
	
//	if(!ap.exists("-d"))
//	{
//		std::cout << "Missing input file name. Specify with -f=0,1,2,3 --> CSR,ELL,HYB,BSR!!!" << std::endl;
//		exit(1);
//	}
//	int format = ap.getInt("-d");;
	int format = 0;
	
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
//	uint64_t edge_count = count_lines(fname);
//	std::cout << "edge_count: " << edge_count << std::endl;
//	load_tsv(fname);

	//coo_benchmark<uint64_t,double>();
	BenchSPMV<int,double> bspmv(format);
	bspmv.load_coo(fname);
	bspmv.power_method();
	//bspmv.edges_to_coo(edges);
	//power_method_csr<int,double>();
	
	///////////
	//destroy//
	///////////
	cusparse_status = cusparseDestroy(cusparse_handle);
	cusp_util::handle_error(cusparse_status,"destroy handle cusparse");
	cublas_status = cublasDestroy(cublas_handle);
	cublas_util::handle_error(cublas_status,"destroy handle cublas");
	
	return 0;
}