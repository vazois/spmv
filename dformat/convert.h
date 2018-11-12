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
void edges_to_csr2(csr_format<Z,T> &csr, std::vector<edge> &edges)
{
	csr.nnz = edges.size();
	csr.m = 0;

	Z *row_idx = (Z*)malloc(sizeof(Z)*csr.nnz);
	Z *col_idx = (Z*)malloc(sizeof(Z)*csr.nnz);
	T *values = (T*)malloc(sizeof(T)*csr.nnz);
	uint64_t count = 0;
	uint64_t j = 1;

	std::cout << "Creating csr ... " << std::endl;
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

	std::cout << "Copying data to GPU ..." << std::endl;
	cutil::safeMalloc<Z,uint64_t>(&(csr.row_idx),sizeof(Z)*(csr.m+1),"csr row_idx alloc");
	cutil::safeMalloc<Z,uint64_t>(&(csr.col_idx),sizeof(Z)*csr.nnz,"csr col_idx alloc");
	cutil::safeMalloc<T,uint64_t>(&(csr.values),sizeof(T)*csr.nnz,"scr values alloc");

	cutil::safeCopyToDevice<Z,uint64_t>(csr.row_idx, row_idx,sizeof(Z)*(csr.m+1), "csr copy to csr.row_idx");
	cutil::safeCopyToDevice<Z,uint64_t>(csr.col_idx, col_idx,sizeof(Z)*csr.nnz, "csr copy to csr.col_idx");
	cutil::safeCopyToDevice<T,uint64_t>(csr.values, values,sizeof(T)*csr.nnz, "csr copy to csr.values");

	free(row_idx);
	free(col_idx);
	free(values);
}


template<class Z, class T>
void edges_to_csr(csr_format<Z,T> &csr, std::vector<edge> &edges)
{
	csr.m = (idx_bounds.mx - idx_bounds.mn + 1);
	csr.nnz = edges.size();
	Z *row_idx = (Z*)malloc(sizeof(Z)*csr.nnz);
	Z *col_idx = (Z*)malloc(sizeof(Z)*csr.nnz);
	T *values = (T*)malloc(sizeof(T)*csr.nnz);

	//Create coo//
	std::cout << "Creating coo in host ..." << std::endl;
	for(uint64_t i = 0; i < edges.size(); i++)
	{
		row_idx[i] = edges[i].first - idx_bounds.mn;
		col_idx[i] = edges[i].second - idx_bounds.mn;
		values[i] = 1.0f;
	}

	edges.clear();
	coo_format<Z,T> coo;
	std::cout << "Allocating data to GPU ..." << std::endl;
	cutil::safeMalloc<Z,uint64_t>(&(coo.row_idx),sizeof(Z)*coo.nnz,"coo row_idx alloc");//ROW IDX FOR COO
	cutil::safeMalloc<Z,uint64_t>(&(csr.col_idx),sizeof(Z)*coo.nnz,"csr col_idx alloc");//COL IDX FOR CSR
	cutil::safeMalloc<T,uint64_t>(&(csr.values),sizeof(T)*coo.nnz,"csr values alloc");

	std::cout << "Copying data to GPU ..." << std::endl;
	cutil::safeCopyToDevice<Z,uint64_t>(coo.row_idx, row_idx,sizeof(Z)*coo.nnz, "coo copy to coo.row_idx");
	cutil::safeCopyToDevice<Z,uint64_t>(csr.col_idx, col_idx,sizeof(Z)*csr.nnz, "csr copy to csr.col_idx");
	cutil::safeCopyToDevice<T,uint64_t>(csr.values, values,sizeof(T)*csr.nnz, "csr copy to csr.values");

//	std::cout << "Converting data to csr ..." <<std::endl;
//	cusparse_status = cusparseXcoo2csr(cusparse_handle, coo.row_idx, csr.nnz, csr.m, csr.row_idx, CUSPARSE_INDEX_BASE_ZERO);
//	cusp_util::handle_error(cusparse_status,"coo to csr");

	destroy_coo<Z,T>(coo);
	free(row_idx);
	free(col_idx);
	free(values);
}

#endif
