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

	Z *row_idx = (Z*)malloc(sizeof(Z)*coo.nnz);
	Z *col_idx = (Z*)malloc(sizeof(Z)*coo.nnz);
	T *values = (T*)malloc(sizeof(T)*coo.nnz);
	for(uint64_t i = 0; i< coo.nnz;i++)
	{
//		std::cout << edges[i].first << std::endl;
		row_idx[i] = edges[i].first;
		col_idx[i] = edges[i].second;
		values[i] = 0.5;
	}

	cutil::safeCopyToDevice<Z,uint64_t>(coo.row_idx, row_idx,sizeof(Z)*coo.nnz, " copy to coo.row_idx");
	cutil::safeCopyToDevice<Z,uint64_t>(coo.col_idx, col_idx,sizeof(Z)*coo.nnz, " copy to coo.col_idx");
	cutil::safeCopyToDevice<T,uint64_t>(coo.values, values,sizeof(T)*coo.nnz, " copy to coo.values");

	free(row_idx);
	free(col_idx);
	free(values);
}


#endif
