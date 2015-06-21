/// @author		Piotr Gregor
/// @date		20 Jun 2015
/// @brief		Generate fair call price for a given set of European options
///				using Monte Carlo approach.


#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>

// includes, project
#include <cuda_runtime.h>
#include <helper_functions.h> // Helper functions (utilities, parsing, timing)
#include <helper_cuda.h>      // helper functions (cuda error checking and intialization)
#include <multithreading.h>

#include "MonteCarlo_common.h"

int *pArgc = NULL;
char **pArgv = NULL;

#ifdef WIN32
#define strcasecmp _strcmpi
#endif

////////////////////////////////////////////////////////////////////////////////
// Common functions
////////////////////////////////////////////////////////////////////////////////
float randFloat(float low, float high)
{
	float t = (float)rand() / (float)RAND_MAX;
	return (1.0f - t) * low + t * high;
}

/// Utility function to tweak problem size for small GPUs
int adjustProblemSize(int GPU_N, int default_nOptions)
{
	int nOptions = default_nOptions;

	// select problem size
	for (int i = 0; i<GPU_N; i++)
	{
		cudaDeviceProp deviceProp;
		checkCudaErrors(cudaGetDeviceProperties(&deviceProp, i));
		int cudaCores = _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor)
			* deviceProp.multiProcessorCount;

		if (cudaCores <= 32)
		{
			nOptions = (nOptions < cudaCores / 2 ? nOptions : cudaCores / 2);
		}
	}

	return nOptions;
}

int adjustGridSize(int GPUIndex, int defaultGridSize)
{
	cudaDeviceProp deviceProp;
	checkCudaErrors(cudaGetDeviceProperties(&deviceProp, GPUIndex));
	int maxGridSize = deviceProp.multiProcessorCount * 40;
	return ((defaultGridSize > maxGridSize) ? maxGridSize : defaultGridSize);
}

///////////////////////////////////////////////////////////////////////////////
// CPU reference functions
///////////////////////////////////////////////////////////////////////////////
extern "C" void MonteCarloCPU(
	TOptionValue    &callValue,
	const TOptionData optionData,
	const size_t pathN,
	int nsteps,
	int  thread_id
	);

//Black-Scholes formula for call options
extern "C" void BlackScholesCall(
	float &CallResult,
	TOptionData optionData
	);


////////////////////////////////////////////////////////////////////////////////
// GPU-driving host thread
////////////////////////////////////////////////////////////////////////////////
//Timer
StopWatchInterface **hTimer = NULL;



///////////////////////////////////////////////////////////////////////////////
// Main program
///////////////////////////////////////////////////////////////////////////////
#define DO_CPU
//#undef DO_CPU

#define DO_GPU
#undef DO_GPU

#define MAX_THREADS 4
TOptionValue partial_outcomes[MAX_THREADS];

#define RANDOMIZE_CALL_PARAMETERS
#undef RANDOMIZE_CALL_PARAMETERS

#define PRINT_RESULTS
#undef PRINT_RESULTS

struct thread_data {
	TOptionValue				optionValue;
	TOptionData					optionData;
	size_t						PATH_N;
	int							nsteps;
	int							thread_id;
	int							OPT_N;
};

thread_data* pDataArray[4];
DWORD   dwThreadIdArray[4];
HANDLE  hThreadArray[4];

void usage()
{
	printf("--method=[threaded,streamed] --scaling=[strong,weak] [--help]\n");
	printf("Method=threaded: 1 CPU thread for each GPU     [default]\n");
	printf("       streamed: 1 CPU thread handles all GPUs (requires CUDA 4.0 or newer)\n");
	printf("Scaling=strong : constant problem size\n");
	printf("        weak   : problem size scales with number of available GPUs [default]\n");
}

DWORD WINAPI thread_f(LPVOID lpParam)
{
	//
	HANDLE							hStdout;
	thread_data						*pData;

	// Make sure there is a console to receive output results. 
	hStdout = GetStdHandle(STD_OUTPUT_HANDLE);
	if (hStdout == INVALID_HANDLE_VALUE)
		return 1;

	// Cast the parameter to the correct data type.
	// The pointer is known to be valid because 
	// it was checked for NULL before the thread was created.
	pData = (thread_data*)lpParam;

	// generate PATH_N outcome prices
	MonteCarloCPU(
		pData->optionValue,
		pData->optionData,
		pData->PATH_N,
		pData->nsteps,
		pData->thread_id
	);
	partial_outcomes[pData->thread_id].callExpected = pData->optionValue.callExpected;
	partial_outcomes[pData->thread_id].putExpected = pData->optionValue.putExpected;
	partial_outcomes[pData->thread_id].callConfidence = pData->optionValue.callConfidence;
	partial_outcomes[pData->thread_id].putConfidence = pData->optionValue.putConfidence;
	return 0;
}

int main(int argc, char **argv)
{
	char *multiMethodChoice = NULL;
	char *scalingChoice = NULL;
	bool use_threads = true;
	bool bqatest = false;
	bool strongScaling = false;

	//GPU number present in the system
	int GPU_N;
	checkCudaErrors(cudaGetDeviceCount(&GPU_N));
	int nOptions = 1;
	// select problem size
	int scale = (strongScaling) ? 1 : GPU_N;
	int OPT_N = nOptions * scale;
	size_t PATH_N = 7000000;
	int nsteps = 1;

	nOptions = adjustProblemSize(GPU_N, nOptions);

	// initialize the timers
	hTimer = new StopWatchInterface*[GPU_N];

	for (int i = 0; i<GPU_N; i++)
	{
		sdkCreateTimer(&hTimer[i]);
		sdkResetTimer(&hTimer[i]);
	}

	//Input data array
	TOptionData  *optionData = new TOptionData[OPT_N];
	//Final GPU MC results
	TOptionValue *callValueGPU = new TOptionValue[OPT_N];
	//"Theoretical" call values by Black-Scholes formula
	float *callValueBS = new float[OPT_N];

	float t;
	double sumDelta, sumRef;

	printf("Generating input data...");
	srand((unsigned)time(NULL));

	for (int i = 0; i < OPT_N; i++)
	{
#ifdef RANDOMIZE_CALL_PARAMETERS
		optionData[i].S = randFloat(5.0f, 50.0f);
		optionData[i].X = randFloat(10.0f, 25.0f);
		optionData[i].T = randFloat(1.0f, 5.0f);
		optionData[i].R = 0.06f;
		optionData[i].V = 0.10f;
		callValueGPU[i].Expected = -1.0f;
		callValueGPU[i].Confidence = -1.0f;
#else
		optionData[i].S = 100.0f;
		optionData[i].X = 104.0f;
		optionData[i].T = 1.0f;
		optionData[i].R = 0.05f;
		optionData[i].V = 0.10f;
		callValueGPU[i].callExpected = -1.0f;
		callValueGPU[i].callConfidence = -1.0f;
		callValueGPU[i].putExpected = -1.0f;
		callValueGPU[i].putConfidence = -1.0f;
#endif
	}
	printf("  done.\n");

#ifdef DO_CPU

	printf("Running CPU MonteCarlo using [%d] threads...\n", MAX_THREADS);
	printf("Number of options: [%d], \tpaths : [%llu], \tsteps: [%d {law of huge numbers}]\n", OPT_N, (unsigned long long)PATH_N, 1);
	sdkStartTimer(&hTimer[0]);
	checkCudaErrors(cudaSetDevice(0));

	TOptionValue optionValue = {};
	sumDelta = 0;
	sumRef = 0;
	double call_price_sum = 0;
	double put_price_sum = 0;

	for (int i = 0; i < OPT_N; i++)
	{
		for (int j = 0; j< MAX_THREADS; j++)
		{
			// Allocate memory for thread data.
			pDataArray[j] = (thread_data*)HeapAlloc(GetProcessHeap(), HEAP_ZERO_MEMORY,
				sizeof(thread_data));

			if (pDataArray[j] == NULL)
			{
				// If the array allocation fails, the system is out of memory
				// so there is no point in trying to print an error message.
				// Just terminate execution.
				ExitProcess(2);
			}

			// Generate unique data for each thread to work with.
			pDataArray[j]->optionValue = optionValue;
			pDataArray[j]->optionData = *optionData;		// bitwise copy
			pDataArray[j]->PATH_N = PATH_N / MAX_THREADS;	// generate fraction of total paths
			pDataArray[j]->nsteps = nsteps;
			pDataArray[j]->thread_id = j;
			pDataArray[j]->OPT_N = OPT_N;

			// Create the thread to begin execution on its own.
			hThreadArray[j] = CreateThread(
				NULL,                   // default security attributes
				0,                      // use default stack size  
				thread_f,			    // thread function name
				pDataArray[j],          // argument to thread function 
				0,                      // use default creation flags 
				&dwThreadIdArray[j]);   // returns the thread identifier 

			// Check the return value for success.
			// If CreateThread fails, terminate execution. 
			// This will automatically clean up threads and memory. 
			if (hThreadArray[j] == NULL)
			{
				ExitProcess(3);
			}
		} // End of main thread creation loop.

		// Wait until all threads have terminated.
		WaitForMultipleObjects(MAX_THREADS, hThreadArray, TRUE, INFINITE);

		// Close all thread handles and free memory allocations.
		for (int k = 0; k < MAX_THREADS; k++)
		{
			CloseHandle(hThreadArray[k]);
			if (pDataArray[k] != NULL)
			{
				HeapFree(GetProcessHeap(), 0, pDataArray[k]);
				pDataArray[k] = NULL;    // Ensure address is not reused.
			}
		}
		/////////////////

		// aggregate outcomes
		double call_expected = 0.0;
		for (int k = 0; k < MAX_THREADS; k++)
		{
			call_expected += partial_outcomes[k].callExpected;
		}
		call_expected /= MAX_THREADS;
		double call_conf = 0.0;
		for (int k = 0; k < MAX_THREADS; k++)
		{
			call_conf += partial_outcomes[k].callConfidence;
		}
		call_conf /= MAX_THREADS;
		double put_expected = 0.0;
		for (int k = 0; k < MAX_THREADS; k++)
		{
			put_expected += partial_outcomes[k].putExpected;
		}
		put_expected /= MAX_THREADS;
		double put_conf = 0.0;
		for (int k = 0; k < MAX_THREADS; k++)
		{
			put_conf += partial_outcomes[k].putConfidence;
		}
		put_conf /= MAX_THREADS;

		printf("\n[%d] Done.\n", i + 1);
		printf("[%d] Call expected : %f\t", i + 1, call_expected);
		printf("Call confidence width: %f [1 - alpha = 0.95]\n", call_conf);
		printf("[%d] Put expected  : %f\t", i + 1, put_expected);
		printf("Put confidence width : %f [1 - alpha = 0.95]\n", put_conf);
		call_price_sum += call_expected;
		put_price_sum += put_expected;
	}
	t = sdkGetTimerValue(&hTimer[0]);
	printf("All done.\nAverage price : \tCall [%f], \tPut [%f]\n\n", call_price_sum / OPT_N, put_price_sum / OPT_N);
	printf("Total time (ms.): %f\n", t);
	//printf("L1 norm: %E\n", sumDelta / sumRef);
#endif

	printf("Shutting down...\n");

	delete[] callValueBS;
	delete[] callValueGPU;
	delete[] optionData;
}