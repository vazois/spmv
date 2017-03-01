#ifndef INIT_CONFIG_H
#define INIT_CONFIG_H

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

double randValue(){
	double X=((double)rand()*2/(double)RAND_MAX) + 1.0f;
	return roundf(X);
}

void init(double *&A, double *&B, uint64_t M, uint64_t N, uint64_t K){
	srand(time(NULL));
	for(uint64_t i = 0; i< M*N; i++) A[i] = randValue();
	for(uint64_t i = 0; i< N*K; i++) B[i] = randValue();
}

float randValueF(){
	float X=(float)rand()/(float)(RAND_MAX) + 1.0f;
	return roundf(X);
}

void initF(float *&A, float *&B, uint64_t N){
	srand(time(NULL));
	for(uint64_t i = 0 ; i < N*N;i++){
		A[i] = randValueF();
		B[i] = randValueF();
	}
}

void zeros(double *&C, uint64_t M, uint64_t K){
	for(unsigned int i = 0 ; i < M*K;i++){ C[i] = 0; }
}



#endif
