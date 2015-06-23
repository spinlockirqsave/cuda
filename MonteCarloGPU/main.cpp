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


int main(int argc, char **argv) {
	try {
		if (argc < 2)
		{
			printf("This program takes one parameter as input. Integer value 0 [GPU + CPU], 1 [GPU]\n");
			exit(-1);
		}
		int DO_CPU;
		sscanf(argv[1], "%d", &DO_CPU);
		DO_CPU ^= 1;
		// declare variables and constants
		const size_t N_PATHS = 7000000;
		const int N_STEPS = 1;
		const size_t N_NORMALS = N_PATHS*N_STEPS;
		const int OPT_N = 1;

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

		// GPU
		double call_sum = 0.0, call_sum2 = 0;
		double call_price = 0.0;
		double call_price_sum = 0.0, call_price_sum2 = 0.0;
		double call_conf = 0.0;
		double call_conf_sum = 0.0;

		double put_sum = 0.0, put_sum2 = 0;
		double put_price = 0.0;
		double put_price_sum = 0.0, put_price_sum2 = 0.0;
		double put_conf = 0.0;
		double put_conf_sum = 0.0;
		// CPU
		double call_sum_cpu = 0.0, call_sum2_cpu = 0;
		double call_price_cpu = 0.0;
		double call_price_cpu_sum = 0.0;
		double call_conf_cpu = 0.0;
		double call_conf_sum_cpu = 0.0;
		double put_sum_cpu = 0.0, put_sum2_cpu = 0;
		double put_price_cpu = 0.0;
		double put_price_cpu_sum = 0.0;
		double put_conf_cpu = 0.0;
		double put_conf_sum_cpu = 0.0;
		//time
		double t2, t4, t5;

		for (int i = 0; i < OPT_N; ++i)
		{
			// generate arrays
			vector<TOptionValue> payoffs(N_PATHS);
			dev_array<TOptionValue> d_payoffs(N_PATHS);
			dev_array<real> d_normals(N_NORMALS);

			// generate random numbers
			printf("Generating random variables...");
			curandGenerator_t curandGenerator;
			curandCreateGenerator(&curandGenerator, CURAND_RNG_PSEUDO_MTGP32);
			curandSetPseudoRandomGeneratorSeed(curandGenerator, time(NULL));//1234ULL);
			curandGenerateNormal(curandGenerator, (float*)d_normals.getData(), N_NORMALS, 0.0f, 1.0f); // n has to be even
			printf(" done.\n");
			printf("Number of options: [%d], \tpaths : [%llu], \tsteps: [%d {law of huge numbers}]\n", 1, (unsigned long long)N_PATHS, 1);
			printf("Running GPU MonteCarlo...\n");
			t2 = double(clock()) / CLOCKS_PER_SEC;

			// call the kernel
			mc_call(d_payoffs.getData(), T, K, S0, r, dt, d_normals.getData(), MuByT, VBySqrtT, N_STEPS, N_PATHS, mu, sigma);
			cudaDeviceSynchronize();

			// copy results from device to host
			d_payoffs.get(&payoffs[0], N_PATHS);

			// aggregate outcomes
			// compute the payoff average
			// Call
			for (size_t j = 0; j < N_PATHS; j++) {
				call_sum += payoffs[j].callExpected;
				call_sum2 += payoffs[j].callExpected * payoffs[j].callExpected;
			}
			call_price = call_sum / N_PATHS;
			call_sum = call_sum * exp(r*T);
			call_sum2 = call_sum2 * exp(r*T) * exp(r*T);
			//Standard deviation
			real call_stdDev = sqrt(((double)N_PATHS * call_sum2 - call_sum * call_sum) / ((double)N_PATHS * (double)(N_PATHS - 1)));
			//Confidence width; in 95% of all cases theoretical value lies within these borders
			call_conf = (float)(exp(-r * T) * 1.96 * call_stdDev / sqrt((double)N_PATHS));

			// Put
			for (size_t j = 0; j < N_PATHS; j++) {
				put_sum += payoffs[j].putExpected;
				put_sum2 += payoffs[j].putExpected * payoffs[j].putExpected;
			}
			put_price = put_sum / N_PATHS;
			put_sum = put_sum * exp(r*T);
			put_sum2 = put_sum2 * exp(r*T) * exp(r*T);
			//Standard deviation
			real put_stdDev = sqrt(((double)N_PATHS * put_sum2 - put_sum * put_sum) / ((double)N_PATHS * (double)(N_PATHS - 1)));
			//Confidence width; in 95% of all cases theoretical value lies within these borders
			put_conf = (float)(exp(-r * T) * 1.96 * put_stdDev / sqrt((double)N_PATHS));

			t4 = double(clock()) / CLOCKS_PER_SEC;
			t5 = t4;
			double ticks = t4;
			printf("[%d]Done.\n", i + 1);
			printf("[%d] Call expected : %.4f\t", i + 1, call_price);
			printf("Call confidence width: %.4f [1 - alpha = 0.95]\n", call_conf);
			printf("[%d] Put expected  : %.4f\t", i + 1, put_price);
			printf("Put confidence width : %.4f [1 - alpha = 0.95]\n", put_conf);

			if (DO_CPU)
			{
				// CPU
				printf("Running CPU MonteCarlo using GPU-preallocated random variables...\n.");

				// init variables for CPU Monte Carlo
				vector<real> normals(N_NORMALS);
				d_normals.get(&normals[0], N_NORMALS);

				double s_curr = 0.0;
				// CPU Monte Carlo Simulation
				for (size_t j = 0; j < N_PATHS; j++) {
					size_t n_idx = j * N_STEPS;
					s_curr = S0;
					int n = 0;
					//do {
					//	//s_curr = s_curr + mu*s_curr*dt + sigma*s_curr*normals[n_idx];
					//	s_curr = s_curr * exp(MuByT + VBySqrtT *  normals[n_idx]);
					//	n_idx++;
					//	n++;
					//} while (n < N_STEPS);
					s_curr = s_curr * exp((r - 0.5 * sigma * sigma) * T + sigma * sqrt(T) * normals[n_idx]);
					double call_payoff = (s_curr > K ? s_curr - K : 0.0);
					call_price_cpu += exp(-r*T) * call_payoff;
					double put_payoff = (s_curr < K ? K - s_curr : 0.0);
					put_price_cpu += exp(-r*T) * put_payoff;
					call_sum_cpu += call_payoff;
					call_sum2_cpu += call_payoff * call_payoff;
					put_sum_cpu += put_payoff;
					put_sum2_cpu += put_payoff * put_payoff;

					if (((double)clock() / CLOCKS_PER_SEC - ticks) > 0.1)
					{
						printf(".");
						ticks = (double)clock() / CLOCKS_PER_SEC;
					}
				}

				call_price_cpu /= N_PATHS;
				put_price_cpu /= N_PATHS;
				// Call standard deviation
				real call_stdDev_cpu = sqrt(((double)N_PATHS * call_sum2_cpu - call_sum_cpu * call_sum_cpu) / ((double)N_PATHS * (double)(N_PATHS - 1)));
				//Confidence width; in 95% of all cases theoretical value lies within these borders
				call_conf_cpu = (float)(exp(-r * T) * 1.96 * call_stdDev_cpu / sqrt((double)N_PATHS));
				// Put standard deviation
				real put_stdDev_cpu = sqrt(((double)N_PATHS * put_sum2_cpu - put_sum_cpu * put_sum_cpu) / ((double)N_PATHS * (double)(N_PATHS - 1)));
				//Confidence width; in 95% of all cases theoretical value lies within these borders
				put_conf_cpu = (float)(exp(-r * T) * 1.96 * put_stdDev_cpu / sqrt((double)N_PATHS));
				t5 = double(clock()) / CLOCKS_PER_SEC;

				printf("\n[%d] Done.\n", i + 1);
				printf("[%d] Call expected : %.4f\t", i + 1, call_price_cpu);
				printf("Call confidence width: %.4f [1 - alpha = 0.95]\n", call_conf_cpu);
				printf("[%d] Put expected  : %.4f\t", i + 1, put_price_cpu);
				printf("Put confidence width : %.4f [1 - alpha = 0.95]\n", put_conf_cpu);

				call_price_cpu_sum += call_price_cpu;
				put_price_cpu_sum += put_price_cpu;
				call_conf_sum_cpu += call_conf_cpu;
				put_conf_sum_cpu += put_conf_cpu;
			}
			// destroy generator
			curandDestroyGenerator(curandGenerator);

			// averages
			call_price_sum += call_price;
			put_price_sum += put_price;
			call_conf_sum += call_conf;
			put_conf_sum += put_conf;
		}

		printf("\nAll done.\n");
		printf("Total time (ms.): %f\n", (t5 - t2)*1e3);
		printf("GPU time: [%6.f] ms", (t4 - t2)*1e3); printf("\t[%4.2f %%]\n", 100 * (t4 - t2) / (t5 - t2));
		if (DO_CPU)
		{
			printf("CPU time: [%6.f] ms", (t5 - t4)*1e3); printf("\t[%4.2f %%]\n", 100 * (t5 - t4) / (t5 - t2));
		}
		printf("Average price :\n");
		printf( "Call price (GPU): %.4f", call_price_sum / OPT_N);      printf("\tCall confidence width: %.4f [1 - alpha = 0.95]\n", call_conf_sum / OPT_N);
		if (DO_CPU)
		{
			printf("Call price (CPU): %.4f", call_price_cpu_sum / OPT_N); printf("\tCall confidence width: %.4f [1 - alpha = 0.95]\n", call_conf_sum_cpu / OPT_N);
		}
		printf( "Put price (GPU) : %.4f", put_price_sum / OPT_N);       printf("\tPut confidence width : %.4f [1 - alpha = 0.95]\n", put_conf_sum / OPT_N);
		if (DO_CPU)
		{
			printf("Put price (CPU) : %.4f", put_price_cpu_sum / OPT_N);   printf("\tPut confidence width : %.4f [1 - alpha = 0.95]\n", put_conf_sum_cpu / OPT_N);
		}
	}
	catch (exception& e) {
		cout << "exception: " << e.what() << "\n";
	}
}