/**
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */


#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#include <curand.h>

//#include "curand_kernel.h"
#include "helper_cuda.h"



////////////////////////////////////////////////////////////////////////////////
// Common types
////////////////////////////////////////////////////////////////////////////////
#include "MonteCarlo_common.h"


////////////////////////////////////////////////////////////////////////////////
// Black-Scholes formula for Monte Carlo results validation
////////////////////////////////////////////////////////////////////////////////
#define A1 0.31938153
#define A2 -0.356563782
#define A3 1.781477937
#define A4 -1.821255978
#define A5 1.330274429
#define RSQRT2PI 0.39894228040143267793994605993438

//Polynomial approxiamtion of
//cumulative normal distribution function
double CND(double d)
{
    double
    K = 1.0 / (1.0 + 0.2316419 * fabs(d));

    double
    cnd = RSQRT2PI * exp(- 0.5 * d * d) *
          (K * (A1 + K * (A2 + K * (A3 + K * (A4 + K * A5)))));

    if (d > 0)
        cnd = 1.0 - cnd;

    return cnd;
}

//Black-Scholes formula for call value
extern "C" void BlackScholesCall(
    float &callValue,
    TOptionData optionData
)
{
    double     S = optionData.S;
    double     X = optionData.X;
    double     T = optionData.T;
    double     R = optionData.R;
    double     V = optionData.V;

    double sqrtT = sqrt(T);
    double    d1 = (log(S / X) + (R + 0.5 * V * V) * T) / (V * sqrtT);
    double    d2 = d1 - V * sqrtT;
    double CNDD1 = CND(d1);
    double CNDD2 = CND(d2);
    double expRT = exp(- R * T);

    callValue = (float)(S * CNDD1 - X * expRT * CNDD2);
}


////////////////////////////////////////////////////////////////////////////////
// CPU Monte Carlo
////////////////////////////////////////////////////////////////////////////////
static double endCallValue(double S, double X, double r, double MuByT, double VBySqrtT)
{
    double callValue = S * exp(MuByT + VBySqrtT * r) - X;
    return (callValue > 0) ? callValue : 0;
}

static double s_t(double S, double r, double MuByT, double VBySqrtT)
{
	return S * exp(MuByT + VBySqrtT * r);
}

extern "C" void MonteCarloCPU(
	TOptionValue    &optionValue,
	const TOptionData optionData,
	const size_t pathN,
	int nsteps,
	int  thread_id
	)
{
	// generrating MC we take advantage of the law of huge numbers
	nsteps = 1;
	const double        T = optionData.T;
	//const double		dt = T / nsteps;
	const double        S = optionData.S;
	const double        X = optionData.X;
	const double        R = optionData.R;
	const double        V = optionData.V;
	//const double    MuByT = (R - 0.5 * V * V) * dt;
	//const double VBySqrtT = V * sqrt(dt);

	float *samples;
	curandGenerator_t gen;

	checkCudaErrors(curandCreateGeneratorHost(&gen, CURAND_RNG_PSEUDO_DEFAULT));
	unsigned long long seed = time(NULL);
	checkCudaErrors(curandSetPseudoRandomGeneratorSeed(gen, seed));


	double call_sum = 0, call_sum2 = 0, st, call_price;
	double put_sum = 0, put_sum2 = 0, put_price;
	double ticks = (double)clock() / CLOCKS_PER_SEC;
	int count_dots = 0;
	// generates pathN outcome prices
	for (size_t pos = 0; pos < pathN; pos++)
	{
		st = optionData.S;
		// curandGenerateNormal requires even number of variables to generate, let's give it 2 then
		samples = (float *)malloc((nsteps + 1) * sizeof(float));
		checkCudaErrors(curandGenerateNormal(gen, samples, (nsteps + 1), 0.0, 1.0));

		//for (int j = 0; j < nsteps; ++j)
		//{
		//	double    sample = samples[j];
		//	st = s_t(st, sample, MuByT, VBySqrtT);
		//}
		st = st * exp((R - 0.5 * V * V) * T + V * sqrt(T) * samples[0]);
		call_price = (st - X > 0 ? st - X : 0);
		put_price = (st - X < 0 ? X - st : 0);
		call_sum += call_price;
		call_sum2 += call_price * call_price;
		put_sum += put_price;
		put_sum2 += put_price * put_price;
		free(samples);
		if (((double)clock() / CLOCKS_PER_SEC - ticks) > 0.3 && thread_id == 0)
		{
			ticks = (double)clock() / CLOCKS_PER_SEC;
			if (++count_dots % 50 == 0)
				printf("\n");
			else
				printf(".");
		}
	}
	//Derive average from the total sum and discount by riskfree rate
	optionValue.callExpected = (float)(exp(-R * T) * call_sum / (double)pathN);
	//Standard deviation
	double call_stdDev = sqrt(((double)pathN * call_sum2 - call_sum * call_sum) / ((double)pathN * (double)(pathN - 1)));
	//Confidence width; in 95% of all cases theoretical value lies within these borders
	optionValue.callConfidence = (float)(exp(-R * T) * 1.96 * call_stdDev / sqrt((double)pathN));

	optionValue.putExpected = (float)(exp(-R * T) * put_sum / (double)pathN);
	double put_stdDev = sqrt(((double)pathN * put_sum2 - put_sum * put_sum) / ((double)pathN * (double)(pathN - 1)));
	optionValue.putConfidence = (float)(exp(-R * T) * 1.96 * put_stdDev / sqrt((double)pathN));

	checkCudaErrors(curandDestroyGenerator(gen));
}
