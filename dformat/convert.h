#ifndef CONVERT_H
#define CONVERT_H

#include "formats.h"
#include "../cuda/CudaHelper.h"

template<class Z, class T>
void edges_to_coo(coo_format<Z,T> &coo,std::vector<edge> &edges)
{
	coo.nnz = edges.size();
	cutil::safeMalloc<Z,uint64_t>(&(coo.row_idx),sizeof(Z)*coo.nnz,"row_idx alloc");
	cutil::safeMalloc<Z,uint64_t>(&(coo.col_idx),sizeof(Z)*coo.nnz,"col_idx alloc");
	cutil::safeMalloc<T,uint64_t>(&(coo.values),sizeof(T)*coo.nnz,"values alloc");


}


#endif
