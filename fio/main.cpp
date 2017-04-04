#include "mmio.h"
//#include <stdint.h>
#include "../tools/ArgParser.h"

int main(int argc, char **argv){
	ArgParser ap;
	ap.parseArgs(argc,argv);
	std::string fpath=ap.getString("-f");
	std::cout<<"fpath: "<<fpath<<std::endl;
	MM_typecode matcode;

	FILE *fp;
	if ((fp = fopen(fpath.c_str(), "r")) == NULL) exit(1);

	if (mm_read_banner(fp, &matcode) != 0){
		printf("Could not process Matrix Market banner.\n");
		exit(1);
	}


	uint32_t nnz = 0;
	uint32_t m, n;
	uint32_t status;
	if ((status = mm_read_mtx_crd_size(fp, &m, &n, &nnz)) !=0) exit(1); //  m=rows, n=cols, nz = non zeros

	uint32_t *rows = (uint32_t*) malloc(sizeof(uint32_t)*nnz);
	uint32_t *cols = (uint32_t*) malloc(sizeof(uint32_t)*nnz);
	double *val = (double*) malloc(sizeof(double)*nnz);

	if (fp !=stdin) fclose(fp);
	return 0;
}
