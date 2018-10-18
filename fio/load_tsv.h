#ifndef LOAD_TSV_H
#define LOAD_TSV_H

#include "../dformat/formats.h"
#include <algorithm>

uint64_t count_lines(std::string fname)
{
	uint64_t count = 0;
	FILE *fp = fopen(fname.c_str(), "r");

	uint64_t size = 1024;
	char *buffer = (char*)malloc(sizeof(char)*size);
	int bytes = 0;

	do{//Count N
		for(int i = 0; i < bytes; i++) if( buffer[i] == '\n') count++;
	}while( (bytes = fread(buffer,sizeof(char),size, fp)) > 0 );

	fclose(fp);
	free(buffer);
	return count;
}

void load_tsv(std::string fname)
{
	FILE *fp = fopen(fname.c_str(), "r");
	uint64_t n[2] = {0,0};

	uint64_t mx[2] = {0,0};
	uint64_t mn[2] = {ULONG_MAX,ULONG_MAX};
	std::cout << "Loading edges from " << fname << " !!!" <<std::endl;
	while(fscanf(fp,"%ld %ld",&n[0],&n[1]) > 0)
	{
		//std::cout << n[0] << "-->" << n[1] << std::endl;
		mx[0] = std::max(n[0],mx[0]); mx[1] = std::max(n[1],mx[1]);
		mn[0] = std::min(n[0],mn[0]); mn[1] = std::min(n[1],mn[1]);

		edge e;
		e.first = n[0];
		e.second = n[1];
		edges.push_back(e);
	};
	std::cout << "max: " << mx[0] << "," << mx[1] << std::endl;
	std::cout << "min: " << mn[0] << "," << mn[1] << std::endl;
	std::sort(edges.begin(),edges.end(),sort_edge_asc);
	//for(uint64_t i = 0; i < 10; i++) std::cout << edges[i].first << "--->" << edges[i].second << std::endl;
	idx_bounds.mx = std::max(mx[0],mx[1]);
	idx_bounds.mn = std::max(mn[0],mn[1]);
	fclose(fp);
}

#endif
