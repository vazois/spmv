#ifndef CUBLAS_HELPER_H
#define CUBLAS_HELPER_H

#include "../cuda/CudaHelper.h"
#include <cublas_v2.h>

static cublasStatus_t cublas_status;
static cublasHandle_t cublas_handle;

namespace cublas_util{

	void handle_error(cublasStatus_t &status, std::string msg)
	{
		cutil::cudaCheckErr(cudaDeviceSynchronize(),msg);
		switch(status){
			case CUBLAS_STATUS_SUCCESS:
				break;
			case CUBLAS_STATUS_NOT_INITIALIZED:
				std::cout<< "(ERROR) at {"<<msg<<"} Status not initialized!!!" << std::endl;
				break;
			case CUBLAS_STATUS_ALLOC_FAILED:
				std::cout<< "(ERROR) at {"<<msg<<"} Status alloc failure!!!" << std::endl;
				break;
			case CUBLAS_STATUS_INVALID_VALUE:
				std::cout<< "(ERROR) at {"<<msg<<"} Invalid value!!!" << std::endl;
				break;
			case CUBLAS_STATUS_ARCH_MISMATCH:
				std::cout<< "(ERROR) at {"<<msg<<"} Architecture mismatch!!!" << std::endl;
				break;
			case CUBLAS_STATUS_MAPPING_ERROR:
				std::cout<< "(ERROR) at {"<<msg<<"} Mapping error!!!" << std::endl;
				break;
			case CUBLAS_STATUS_EXECUTION_FAILED:
				std::cout<< "(ERROR) at {"<<msg<<"} Execution failure!!!" << std::endl;
				break;
			case CUBLAS_STATUS_INTERNAL_ERROR:
				std::cout<< "(ERROR) at {"<<msg<<"} Internal error!!!" << std::endl;
				break;
			case CUBLAS_STATUS_NOT_SUPPORTED:
				std::cout<< "(ERROR) at {"<<msg<<"} Matrix type not supported!!!" << std::endl;
				break;
			default:
				std::cout << "(ERROR)" << std::endl;
		}

	}
};



#endif
