#ifndef FORMATS_H
#define FORMATS_H

#include "../cuda/CudaHelper.h"
#include "../cuda/CuSparseHelper.h"

struct idx_mx_mn
{
	uint64_t mx;
	uint64_t mn;
};
static idx_mx_mn idx_bounds;

struct edge{
	uint64_t first;
	uint64_t second;
};
static std::vector<edge> edges;

template<class Z, class T>
struct coo_format
{
	Z nnz;

	Z *row_idx = NULL;
	Z *col_idx = NULL;
	T *values = NULL;
};

template<class Z,class T>
struct csr_format
{
	Z nnz;
	Z m;

	Z *row_idx = NULL;
	Z *col_idx = NULL;
	T *values = NULL;
};

static bool	sort_edge_asc(const edge &a, const edge &b)
{
	if(a.first == b.first)
		return a.second < b.second;
	else
		return a.first < b.first;
};

template<class Z, class T>
void destroy_coo(coo_format<Z,T> &coo)
{
	if(coo.row_idx != NULL) cudaFree(coo.row_idx);
	if(coo.col_idx != NULL) cudaFree(coo.col_idx);
	if(coo.values != NULL) cudaFree(coo.values);

	coo.row_idx = NULL;
	coo.col_idx = NULL;
	coo.values = NULL;
}

template<class Z, class T>
void destroy_csr(csr_format<Z,T> &csr)
{
	if(csr.row_idx != NULL) cudaFree(csr.row_idx);
	if(csr.col_idx != NULL) cudaFree(csr.col_idx);
	if(csr.values != NULL) cudaFree(csr.values);

	csr.row_idx = NULL;
	csr.col_idx = NULL;
	csr.values = NULL;
}


static cusparseHandle_t cusparse_handle;
static cusparseStatus_t cusparse_status;



#endif
