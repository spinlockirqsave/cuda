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

#ifndef MONTECARLO_COMMON_H
#define MONTECARLO_COMMON_H
#include "realtype.h"
#include "curand_kernel.h"

////////////////////////////////////////////////////////////////////////////////
// Global types
////////////////////////////////////////////////////////////////////////////////
typedef struct
{
    float S;
    float X;
    float T;
    float R;
    float V;
} TOptionData;

typedef struct
        //#ifdef __CUDACC__
        //__align__(8)
        //#endif
{
    float callExpected;
	float putExpected;
    float callConfidence;
	float putConfidence;
} TOptionValue;

//GPU outputs before CPU postprocessing
typedef struct
{
	real callExpected;
	real putExpected;
	real callConfidence;
	real putConfidence;
} __TOptionValue;

#endif
