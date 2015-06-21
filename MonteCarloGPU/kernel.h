#ifndef _KERNEL_CUH_
#define _KERNEL_CUH_

typedef struct
{
	float callExpected;
	float putExpected;
	float callConfidence;
	float putConfidence;
} TOptionValue;

void mc_dao_call(TOptionValue * d_s, float T, float K, float S0, float r, float dt, real* d_normals, const real MuByT, const real VBySqrtT, unsigned N_STEPS, size_t N_PATHS, const real mu, const real V);
#endif