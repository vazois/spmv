#ifndef LOAD_TSV_H
#define LOAD_TSV_H

#include "../dformat/formats.h"
#include <algorithm>

#define BYTES_FRAME 4*1024*1024

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
	long bytes_total = 0;
	long bytes_read = 0;
	long bytes_progress = BYTES_FRAME;
	if(fp == NULL){ perror(("error opening file " + fname).c_str()); exit(1); }
	fseek(fp,0,SEEK_END);
	bytes_total = ftell(fp);
	std::cout << "total(bytes): " << bytes_total << std::endl;
	rewind(fp);


	uint32_t n[2] = {0,0};
	uint32_t mx[2] = {0,0};
	uint32_t mn[2] = {UINT_MAX,UINT_MAX};
	std::cout << "Loading edges from " << fname << " !!!" <<std::endl;
	std::cout << "Loading: [" << (uint32_t)((((double)bytes_read)/(bytes_total))*100) << "]\r";
	std::cout.flush();
	while(fscanf(fp,"%u\t%u",&n[0],&n[1]) > 0)
	{
		//std::cout << n[0] << "-->" << n[1] << std::endl;
		mx[0] = std::max(n[0],mx[0]); mx[1] = std::max(n[1],mx[1]);
		mn[0] = std::min(n[0],mn[0]); mn[1] = std::min(n[1],mn[1]);

		edge e;
		e.first = n[0];
		e.second = n[1];
		edges.push_back(e);

		bytes_read = ftell(fp);
		uint32_t p = bytes_read / bytes_progress;
		if( p > 0 )
		{
			//std::cout << "Loading: [" << (uint32_t)((((double)bytes_read)/(bytes_total))*100) << "] -- " << edges.size() <<" \r";
			std::cout << "Loading: [" << (uint32_t)((((double)bytes_read)/(bytes_total))*100) << "] \r";
			std::cout.flush();
			bytes_progress += BYTES_FRAME;
		}
	};
	std::cout.flush();
	std::cout << "max: " << mx[0] << "," << mx[1] << std::endl;
	idx_bounds.mx = std::max(mx[0],mx[1]);
	idx_bounds.mn = std::min(mn[0],mn[1]);
	fclose(fp);
}

#endif
