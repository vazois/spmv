#include "tools/ArgParser.h"

#include "fio/load_tsv.h"
#include "dformat/convert.h"

template<class Z, class T>
void coo_benchmark()
{
	coo_format<Z,T> coo;
	edges_to_coo<Z,T>(coo,edges);
	
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
	
	/////////////
	//Load File//
	load_tsv(fname);

	coo_benchmark<uint64_t,double>();
	

	return 0;
}