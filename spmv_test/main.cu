#include "../time/Time.h"
#include "../tools/ArgParser.h"


#include "spmv_s_driver.h"

int main(int argc, char **argv){
	ArgParser ap;
	ap.parseArgs(argc,argv);
	small_test_coo();
	return 0;
}
