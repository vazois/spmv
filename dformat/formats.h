#ifndef FORMATS_H
#define FORMATS_H

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

static bool	sort_edge_asc(const edge &a, const edge &b)
{
	if(a.first == b.first)
		return a.second < b.second;
	else
		return a.first < b.first;
};

template<class Z, class T>
void free_coo(coo_format<Z,T> &coo)
{
	if(coo.row_idx != NULL) cudaFree(coo.row_idx);
}


#endif
