#ifndef CU_SPARSE_HELPER_H
#define CU_SPARSE_HELPER_H

#include "../cuda/CudaHelper.h"
#include <iostream>
#include <string>
#include <cusparse_v2.h>

static cusparseHandle_t cusparse_handle;
static cusparseStatus_t cusparse_status;
static cusparseMatDescr_t cusparse_descrA;

static cusparseHybMat_t hybA;

namespace cusp_util{

	void handle_error(cusparseStatus_t &status, std::string msg){
		cutil::cudaCheckErr(cudaDeviceSynchronize(),msg);
		switch(status){
			case CUSPARSE_STATUS_SUCCESS:
				break;
			case CUSPARSE_STATUS_NOT_INITIALIZED:
				std::cout<< "(ERROR) at {"<<msg<<"} Status not initialized!!!" << std::endl;
				break;
			case CUSPARSE_STATUS_ALLOC_FAILED:
				std::cout<< "(ERROR) at {"<<msg<<"} Status alloc failure!!!" << std::endl;
				break;
			case CUSPARSE_STATUS_INVALID_VALUE:
				std::cout<< "(ERROR) at {"<<msg<<"} Invalid value!!!" << std::endl;
				break;
			case CUSPARSE_STATUS_ARCH_MISMATCH:
				std::cout<< "(ERROR) at {"<<msg<<"} Architecture mismatch!!!" << std::endl;
				break;
			case CUSPARSE_STATUS_MAPPING_ERROR:
				std::cout<< "(ERROR) at {"<<msg<<"} Mapping error!!!" << std::endl;
				break;
			case CUSPARSE_STATUS_EXECUTION_FAILED:
				std::cout<< "(ERROR) at {"<<msg<<"} Execution failure!!!" << std::endl;
				break;
			case CUSPARSE_STATUS_INTERNAL_ERROR:
				std::cout<< "(ERROR) at {"<<msg<<"} Internal error!!!" << std::endl;
				break;
			case CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED:
				std::cout<< "(ERROR) at {"<<msg<<"} Matrix type not supported!!!" << std::endl;
				break;
			default:
				std::cout << "(ERROR)" << std::endl;
		}

	}
}

#endif
