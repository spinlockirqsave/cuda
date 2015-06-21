
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "realtype.h"

#include <stdio.h>
#include <stdlib.h>
#include <stdio.h>
#include <helper_cuda.h>
#include <curand_kernel.h>
#include "kernel.h"


__global__ void mc_kernel(
	TOptionValue * d_s,
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
	real call_price = 0.0;
	real put_price = 0.0;
	if (s_idx < N_PATHS) {
		//do {
		//	//s_curr = s_curr + mu*s_curr*dt + sigma*s_curr*d_normals[n_idx];
		//	//s_curr = s_curr * exp(MuByT + VBySqrtT * d_normals[n_idx]);
		//	s_curr = (s_curr * exp( mu*dt + V * sqrt(dt) * d_normals[n_idx]));
		//	n_idx++;
		//	n++;
		//} while (n < N_STEPS);
		s_curr = s_curr * exp((r - 0.5 * V * V) * T + V * sqrt(T) * d_normals[n_idx]);
		//double payoff = 4.5230 * exp(r*T);
		//s_curr = s_curr * exp(mu*T + V*sqrt(T) * d_normals[n_idx]);
		real call_payoff = ((s_curr > K) ? s_curr - K : 0.0);
		real put_payoff = ((s_curr < K) ? K - s_curr : 0.0);
		//__syncthreads();
		// price expectation
		call_price = exp(-r*T) * call_payoff;
		d_s[s_idx].callExpected = call_price;

		put_price = exp(-r*T) * put_payoff;
		d_s[s_idx].putExpected = put_price;
	}
}

void mc_dao_call(
	TOptionValue * d_s,
	float T,
	float K,
	float S0,
	float r,
	float dt,
	real * d_normals,
	const real MuByT,
	const real VBySqrtT,
	unsigned N_STEPS,
	size_t N_PATHS,
	real mu,
	real V) {
	const unsigned BLOCK_SIZE = 1024;
	const unsigned GRID_SIZE = ceil(float(N_PATHS) / float(BLOCK_SIZE));
	//const double MuByT = mu * dt;
	//const double VBySqrtT = sigma * sqrt(dt);
	//printf("MuByT : [%lf], VBySqrtT : [%lf]\n", MuByT, VBySqrtT);
	mc_kernel << <GRID_SIZE, BLOCK_SIZE >> >(
		d_s, T, K, S0,  MuByT, r, VBySqrtT, d_normals, N_STEPS, N_PATHS, mu, V);
}
