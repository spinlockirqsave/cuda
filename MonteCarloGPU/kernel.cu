
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "realtype.h"

#include <stdio.h>
#include <stdlib.h>
#include <stdio.h>
#include <helper_cuda.h>
#include <curand_kernel.h>


__global__ void mc_kernel(
	real * d_s,
	real T,
	real K,
	real S0,
	real MuByT,
	real r,
	real VBySqrtT,
	real * d_normals,
	unsigned N_STEPS,
	size_t N_PATHS,
	real mu,
	real V)
{
	real s_curr = 0.0;
	const unsigned tid = threadIdx.x;
	const unsigned bid = blockIdx.x;
	const unsigned bsz = blockDim.x;
	size_t s_idx = tid + bid * bsz;
	size_t n_idx = tid + bid * bsz;
	//printf("s_idx: [%u]\t", (unsigned long long)tid);
	s_curr = S0;
	const real dt = float(T) / float(N_STEPS);
	if (s_idx < N_PATHS) {
		int n = 0;
		do {
			//s_curr = s_curr + mu*s_curr*dt + sigma*s_curr*d_normals[n_idx];
			//s_curr = s_curr * exp(MuByT + VBySqrtT * d_normals[n_idx]);
			s_curr = s_curr * exp(mu*dt + V*sqrt(dt) * d_normals[n_idx]);
			n_idx++;
			n++;
		} while (n < N_STEPS);
		//double payoff = 4.5230 * exp(r*T);
		//s_curr = s_curr * exp(mu*T + V*sqrt(T) * d_normals[n_idx]);
		real payoff = (s_curr > K ? s_curr - K : 0.0);
		//__syncthreads();
		d_s[s_idx] = exp(-r*T) * payoff;
	}
}

void mc_dao_call(
	real * d_s,
	float T,
	float K,
	float S0,
	float r,
	float dt,
	real * d_normals,
	const double MuByT,
	const double VBySqrtT,
	unsigned N_STEPS,
	size_t N_PATHS,
	double mu,
	double V) {
	const unsigned BLOCK_SIZE = 1024;
	const unsigned GRID_SIZE = ceil(float(N_PATHS) / float(BLOCK_SIZE));
	//const double MuByT = mu * dt;
	//const double VBySqrtT = sigma * sqrt(dt);
	//printf("MuByT : [%lf], VBySqrtT : [%lf]\n", MuByT, VBySqrtT);
	mc_kernel << <GRID_SIZE, BLOCK_SIZE >> >(
		d_s, T, K, S0,  MuByT, r, VBySqrtT, d_normals, N_STEPS, N_PATHS, mu, V);
}
