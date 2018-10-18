#ifndef CONVERT_H
#define CONVERT_H

#include "formats.h"

template<class Z, class T>
uint64_t edges_to_coo(coo_format<Z,T> &coo,std::vector<edge> &edges)
{
	coo.nnz = edges.size();
	cutil::safeMalloc<Z,uint64_t>(&(coo.row_idx),sizeof(Z)*coo.nnz,"row_idx alloc");
	cutil::safeMalloc<Z,uint64_t>(&(coo.col_idx),sizeof(Z)*coo.nnz,"col_idx alloc");
	cutil::safeMalloc<T,uint64_t>(&(coo.values),sizeof(T)*coo.nnz,"values alloc");

	Z *row_idx = (Z*)malloc(sizeof(Z)*coo.nnz);
	Z *col_idx = (Z*)malloc(sizeof(Z)*coo.nnz);
	T *values = (T*)malloc(sizeof(T)*coo.nnz);
	uint64_t m = 0;
	for(uint64_t i = 0; i< coo.nnz;i++)
	{
//		std::cout << edges[i].first << std::endl;
		row_idx[i] = edges[i].first - idx_bounds.mn;
		col_idx[i] = edges[i].second - idx_bounds.mn;
		values[i] = 0.5;
		if(i > 0 & row_idx[i] != row_idx[i-1]){ m++; }
	}

	cutil::safeCopyToDevice<Z,uint64_t>(coo.row_idx, row_idx,sizeof(Z)*coo.nnz, "coo copy to coo.row_idx");
	cutil::safeCopyToDevice<Z,uint64_t>(coo.col_idx, col_idx,sizeof(Z)*coo.nnz, "coo copy to coo.col_idx");
	cutil::safeCopyToDevice<T,uint64_t>(coo.values, values,sizeof(T)*coo.nnz, "coo copy to coo.values");

	free(row_idx);
	free(col_idx);
	free(values);
	return m;
}

template<class Z, class T>
void edges_to_csr(csr_format<Z,T> &csr, std::vector<edge> &edges)
{
	csr.nnz = edges.size();
	csr.m = 0;
	//cutil::safeMalloc<Z,uint64_t>(&(csr.col_idx),sizeof(Z)*csr.nnz,"csr col_idx alloc");
	//cutil::safeMalloc<T,uint64_t>(&(csr.values),sizeof(T)*csr.nnz,"scr values alloc");

	Z *row_idx = (Z*)malloc(sizeof(Z)*csr.nnz);
	Z *col_idx = (Z*)malloc(sizeof(Z)*csr.nnz);
	T *values = (T*)malloc(sizeof(T)*csr.nnz);
	uint64_t count = 0;
	uint64_t j = 1;

	row_idx[0] = 0;
	for(uint64_t i = 0; i < csr.nnz; i++)
	{

		col_idx[i] = edges[i].second - idx_bounds.mn;
		values[i] = 0.5;

		if(i > 0 & (edges[i].first != edges[i-1].first))
		{
			csr.m++;
			row_idx[j] = count;
			j++;
		}
		count++;
	}
	row_idx[csr.m] = count;
	//cutil::safeMalloc<Z,uint64_t>(&(csr.row_idx),sizeof(Z)*csr.m,"csr row_idx alloc");

	for(uint64_t i = 1; i < 2;i++)
	{
		//std::cout << "i: " <<row_idx[i] << std::endl;
		std::cout << "size: " << row_idx[i] - row_idx[i-1] << std::endl;
		for(uint64_t j = row_idx[i-1]; j < row_idx[i]; j++)
		{
			std::cout << "\t" << edges[j].first << "---" << edges[j].second << std::endl;
		}
	}
	std::cout << "csr.m: " << csr.m << std::endl;

	free(row_idx);
	free(col_idx);
	free(values);
}


#endif
