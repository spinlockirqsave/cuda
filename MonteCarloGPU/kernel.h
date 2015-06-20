#ifndef _KERNEL_CUH_
#define _KERNEL_CUH_

void mc_dao_call(real * d_s, float T, float K, float S0, float r, float dt, real* d_normals, const double MuByT, const double VBySqrtT, unsigned N_STEPS, size_t N_PATHS, double mu, double V);
#endif