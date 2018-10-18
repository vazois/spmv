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
	//coo_format<Z,T> coo;
	//uint64_t m = edges_to_coo<Z,T>(coo,edges);
	//std::cout << "m: " << m << std::endl;
	
	//csr.m = m;
	//coo_to_csr<Z,T>(csr,coo);
	
	edges_to_csr<Z,T>(csr,edges);
	
	//destroy_coo<Z,T>(coo);
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
	cusp_util::handle_error(cusparse_status,"create handle");
	/////////////
	//Load File//
	/////////////
	load_tsv(fname);

	//coo_benchmark<uint64_t,double>();
	csr_benchmark<uint32_t,double>();
	
	
	///////////
	//destroy//
	///////////
	cusparse_status = cusparseDestroy(cusparse_handle);
	cusp_util::handle_error(cusparse_status,"create handle");
	
	return 0;
}