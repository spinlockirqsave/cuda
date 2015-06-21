#include <stdio.h>
#include <vector>
#include <time.h>
#include <math.h>
#include <iostream>
#include <time.h>
#include <cuda_runtime.h>
#include "realtype.h"
#include "kernel.h"
#include "dev_array.h"
#include <curand.h>

using namespace std;

int main() {
	try {
		// declare variables and constants
		const size_t N_PATHS = 7000000;
		const int N_STEPS = 140;
		const size_t N_NORMALS = N_PATHS*N_STEPS;

		const float T = 1.0f;
		float dt = float(T) / float(N_STEPS);
		float sqrdt = sqrt(dt);
		const float K = 104.0f;
		const float B = 95.0f;
		const float S0 = 100.0f;
		const float sigma = 0.1f;
		const float r = 0.05f;
		const double mu = (r - 0.5 * sigma * sigma);
		const double MuByT = mu * dt;
		const double VBySqrtT = sigma * sqrt(dt);

		//printf("MuByT : [%lf], VBySqrtT : [%lf]\n", MuByT, VBySqrtT);

		// generate arrays
		vector<real> payoffs(N_PATHS);
		dev_array<real> d_payoffs(N_PATHS);
		dev_array<real> d_normals(N_NORMALS);

		// generate random numbers
		curandGenerator_t curandGenerator;
		curandCreateGenerator(&curandGenerator, CURAND_RNG_PSEUDO_MTGP32);
		curandSetPseudoRandomGeneratorSeed(curandGenerator, time(NULL));//1234ULL);
		curandGenerateNormal(curandGenerator, (float*)d_normals.getData(), N_NORMALS, 0.0f, 1.0f); // n has to be even

		printf("Number of options: [%d], \tpaths : [%llu], \tsteps: [%d]\n", 1, (unsigned long long)N_PATHS, N_STEPS);
		printf("Running GPU MonteCarlo...\n");
		double t2 = double(clock()) / CLOCKS_PER_SEC;

		// call the kernel
		mc_dao_call(d_payoffs.getData(), T, K, S0, r, dt, d_normals.getData(), MuByT, VBySqrtT, N_STEPS, N_PATHS, mu, sigma);
		cudaDeviceSynchronize();

		// copy results from device to host
		d_payoffs.get(&payoffs[0], N_PATHS);

		// compute the payoff average
		double temp_sum = 0.0;
		for (size_t i = 0; i < N_PATHS; i++) {
			temp_sum += payoffs[i];
		}
		temp_sum /= N_PATHS;
		double t4 = double(clock()) / CLOCKS_PER_SEC;
		double ticks = t4;
		printf("Done.\n");
		printf("Running CPU MonteCarlo using GPU-preallocated random variables...\n.");

		// init variables for CPU Monte Carlo
		vector<real> normals(N_NORMALS);
		d_normals.get(&normals[0], N_NORMALS);
		double sum = 0.0;
		double s_curr = 0.0;

		// CPU Monte Carlo Simulation
		for (size_t i = 0; i < N_PATHS; i++) {
			size_t n_idx = i * N_STEPS;
			s_curr = S0;
			int n = 0;
			//do {
			//	//s_curr = s_curr + mu*s_curr*dt + sigma*s_curr*normals[n_idx];
			//	s_curr = s_curr * exp(MuByT + VBySqrtT *  normals[n_idx]);
			//	n_idx++;
			//	n++;
			//} while (n < N_STEPS);
			s_curr = s_curr * exp((r - 0.5 * sigma * sigma) * T + sigma * sqrt(T) * normals[n_idx]);
			double payoff = (s_curr > K ? s_curr - K : 0.0);
			sum += exp(-r*T) * payoff;
			if (((double)clock() / CLOCKS_PER_SEC - ticks) > 0.1)
			{
				printf(".");
				ticks = (double)clock() / CLOCKS_PER_SEC;
			}
		}

		sum /= N_PATHS;
		double t5 = double(clock()) / CLOCKS_PER_SEC;

		printf("\nDone.\nAverage price : \tCall [%f], \tPut [%f]\n\n", temp_sum, temp_sum);
		printf("Total time (ms.): %f\n", (t4 - t2)*1e3);
		cout << "Call price (GPU): " << temp_sum << "\n";
		cout << "Call price (CPU): " << sum << "\n";
		cout << "GPU time: " << (t4 - t2)*1e3 << " ms\n";
		cout << "CPU time: " << (t5 - t4)*1e3 << " ms\n";

		// destroy generator
		curandDestroyGenerator(curandGenerator);
	}
	catch (exception& e) {
		cout << "exception: " << e.what() << "\n";
	}
}