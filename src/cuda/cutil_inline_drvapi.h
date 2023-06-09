/*
 * Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

#ifndef _CUTIL_INLINE_FUNCTIONS_DRVAPI_H_
#define _CUTIL_INLINE_FUNCTIONS_DRVAPI_H_

#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include "drvapi_error_string.h"
#include "string_helper.h"

#define cutilDrvSafeCallNoSync(err)     __cuSafeCallNoSync  (err, __FILE__, __LINE__)
#define cutilDrvSafeCall(err)           __cuSafeCall        (err, __FILE__, __LINE__)
#define cutilDrvCtxSync()               __cuCtxSync         (__FILE__, __LINE__)
#define cutilDrvCheckMsg(msg)           __cuCheckMsg        (msg, __FILE__, __LINE__)
#define cutilDrvAlignOffset(offset, alignment)  ( offset = (offset + (alignment-1)) & ~((alignment-1)) )

inline void __cuSafeCallNoSync( CUresult err, const char *file, const int line )
{
    if( CUDA_SUCCESS != err) {
        fprintf(stderr, "cuSafeCallNoSync() Driver API error = %04d \"%s\" from file <%s>, line %i.\n",
                err, getCudaDrvErrorString(err), file, line );
        exit(-1);
    }
}
inline void __cuSafeCall( CUresult err, const char *file, const int line )
{
    __cuSafeCallNoSync( err, file, line );
}

inline void __cuCtxSync(const char *file, const int line )
{
    CUresult err = cuCtxSynchronize();
    if( CUDA_SUCCESS != err ) {
        fprintf(stderr, "cuCtxSynchronize() API error = %04d \"%s\" from file <%s>, line %i.\n",
                err, getCudaDrvErrorString(err), file, line );
        exit(-1);
    }
}

#ifndef MIN
#define MIN(a,b) ((a < b) ? a : b)
#endif
#ifndef MAX
#define MAX(a,b) ((a > b) ? a : b)
#endif

inline int _ConvertSMVer2CoresDrvApi(int major, int minor)
{
	typedef struct {
		int SM;
		int Cores;
	} sSMtoCores;

        sSMtoCores nGpuArchCoresPerSM[] =
        { { 0x10,  8 },
          { 0x11,  8 },
          { 0x12,  8 },
          { 0x13,  8 },
          { 0x20, 32 },
          { 0x21, 48 },
          {   -1, -1 }
        };

	int index = 0;
	while (nGpuArchCoresPerSM[index].SM != -1) {
		if (nGpuArchCoresPerSM[index].SM == ((major << 4) + minor) ) {
			return nGpuArchCoresPerSM[index].Cores;
		}
		index++;
	}
	printf("MapSMtoCores undefined SMversion %d.%d!\n", major, minor);
	return -1;
}


inline int cutilDrvGetMaxGflopsDeviceId()
{
    CUdevice current_device = 0, max_perf_device = 0;
    int device_count     = 0, sm_per_multiproc = 0;
    int max_compute_perf = 0, best_SM_arch     = 0;
    int major = 0, minor = 0, multiProcessorCount, clockRate;

    cuInit(0);
    cutilDrvSafeCallNoSync(cuDeviceGetCount(&device_count));

	while ( current_device < device_count ) {
		cutilDrvSafeCallNoSync( cuDeviceComputeCapability(&major, &minor, current_device ) );
		if (major > 0 && major < 9999) {
			best_SM_arch = MAX(best_SM_arch, major);
		}
		current_device++;
	}

	current_device = 0;
	while( current_device < device_count ) {
		cutilDrvSafeCallNoSync( cuDeviceGetAttribute( &multiProcessorCount,
                                                            CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
                                                            current_device ) );
        cutilDrvSafeCallNoSync( cuDeviceGetAttribute( &clockRate,
                                                            CU_DEVICE_ATTRIBUTE_CLOCK_RATE,
                                                            current_device ) );
		cutilDrvSafeCallNoSync( cuDeviceComputeCapability(&major, &minor, current_device ) );

		if (major == 9999 && minor == 9999) {
		    sm_per_multiproc = 1;
		} else {
		    sm_per_multiproc = _ConvertSMVer2CoresDrvApi(major, minor);
		}

		int compute_perf  = multiProcessorCount * sm_per_multiproc * clockRate;
		if( compute_perf  > max_compute_perf ) {
			if ( best_SM_arch > 2 ) {
				if (major == best_SM_arch) {
                    max_compute_perf  = compute_perf;
                    max_perf_device   = current_device;
				}
			} else {
				max_compute_perf  = compute_perf;
				max_perf_device   = current_device;
			}
		}
		++current_device;
	}
	return max_perf_device;
}


inline int cutilDrvGetMaxGflopsGraphicsDeviceId()
{
    CUdevice current_device = 0, max_perf_device = 0;
    int device_count     = 0, sm_per_multiproc = 0;
    int max_compute_perf = 0, best_SM_arch     = 0;
    int major = 0, minor = 0, multiProcessorCount, clockRate;
	int bTCC = 0;
	char deviceName[256];

    cuInit(0);
    cutilDrvSafeCallNoSync(cuDeviceGetCount(&device_count));

	while ( current_device < device_count ) {
		cutilDrvSafeCallNoSync( cuDeviceGetName(deviceName, 256, current_device) );
		cutilDrvSafeCallNoSync( cuDeviceComputeCapability(&major, &minor, current_device ) );

#if CUDA_VERSION >= 3020
		cutilDrvSafeCallNoSync( cuDeviceGetAttribute( &bTCC,  CU_DEVICE_ATTRIBUTE_TCC_DRIVER, current_device ) );
#else
		if (deviceName[0] == 'T') bTCC = 1;
#endif
		if (!bTCC) {
			if (major > 0 && major < 9999) {
				best_SM_arch = MAX(best_SM_arch, major);
			}
		}
		current_device++;
	}

	current_device = 0;
	while( current_device < device_count ) {
		cutilDrvSafeCallNoSync( cuDeviceGetAttribute( &multiProcessorCount,
                                                            CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
                                                            current_device ) );
        cutilDrvSafeCallNoSync( cuDeviceGetAttribute( &clockRate,
                                                            CU_DEVICE_ATTRIBUTE_CLOCK_RATE,
                                                            current_device ) );
		cutilDrvSafeCallNoSync( cuDeviceComputeCapability(&major, &minor, current_device ) );

#if CUDA_VERSION >= 3020
		cutilDrvSafeCallNoSync( cuDeviceGetAttribute( &bTCC,  CU_DEVICE_ATTRIBUTE_TCC_DRIVER, current_device ) );
#else
		if (deviceName[0] == 'T') bTCC = 1;
#endif

		if (major == 9999 && minor == 9999) {
		    sm_per_multiproc = 1;
		} else {
		    sm_per_multiproc = _ConvertSMVer2CoresDrvApi(major, minor);
		}

		if (!bTCC)
		{
			int compute_perf  = multiProcessorCount * sm_per_multiproc * clockRate;
			if( compute_perf  > max_compute_perf ) {
				if ( best_SM_arch > 2 ) {
					if (major == best_SM_arch) {
                        max_compute_perf  = compute_perf;
                        max_perf_device   = current_device;
					}
				} else {
					max_compute_perf  = compute_perf;
					max_perf_device   = current_device;
				}
			}
		}
		++current_device;
	}
	return max_perf_device;
}

inline void __cuCheckMsg( const char * msg, const char *file, const int line )
{
    CUresult err = cuCtxSynchronize();
    if( CUDA_SUCCESS != err) {
		fprintf(stderr, "cutilDrvCheckMsg -> %s", msg);
        fprintf(stderr, "cutilDrvCheckMsg -> cuCtxSynchronize API error = %04d \"%s\" in file <%s>, line %i.\n",
                err, getCudaDrvErrorString(err), file, line );
        exit(-1);
    }
}


inline int cutilDeviceInitDrv(int ARGC, char ** ARGV)
{
    int cuDevice = 0;
    int deviceCount = 0;
    CUresult err = cuInit(0);
    if (CUDA_SUCCESS == err)
        cutilDrvSafeCallNoSync(cuDeviceGetCount(&deviceCount));
    if (deviceCount == 0) {
        fprintf(stderr, "CUTIL DeviceInitDrv error: no devices supporting CUDA\n");
        exit(-1);
    }
    int dev = 0;
    dev = getCmdLineArgumentInt(ARGC, (const char **) ARGV, "device=");
    if (dev < 0) dev = 0;
    if (dev > deviceCount-1) {
		fprintf(stderr, "\n");
		fprintf(stderr, ">> %d CUDA capable GPU device(s) detected. <<\n", deviceCount);
        fprintf(stderr, ">> cutilDeviceInit (-device=%d) is not a valid GPU device. <<\n", dev);
		fprintf(stderr, "\n");
        return -dev;
    }
    cutilDrvSafeCallNoSync(cuDeviceGet(&cuDevice, dev));
    char name[100];
    cuDeviceGetName(name, 100, cuDevice);
    if (checkCmdLineFlag(ARGC, (const char **) ARGV, "quiet") == CUTFalse) {
       printf("> Using CUDA Device [%d]: %s\n", dev, name);
   	}
    return dev;
}


inline CUdevice cutilChooseCudaDeviceDrv(int argc, char **argv, int *p_devID)
{
    CUdevice cuDevice;
    int devID = 0;
    if( checkCmdLineFlag(argc, (const char**)argv, "device") ) {
        devID = cutilDeviceInitDrv(argc, argv);
        if (devID < 0) {
            printf("exiting...\n");
            exit(0);
        }
    } else {
        char name[100];
        devID = cutilDrvGetMaxGflopsDeviceId();
        cutilDrvSafeCallNoSync(cuDeviceGet(&cuDevice, devID));
        cuDeviceGetName(name, 100, cuDevice);
        printf("> Using CUDA Device [%d]: %s\n", devID, name);
    }
    cuDeviceGet(&cuDevice, devID);
    if (p_devID) *p_devID = devID;
    return cuDevice;
}


inline void cutilDrvCudaCheckCtxLost(const char *errorMessage, const char *file, const int line )
{
    CUresult err = cuCtxSynchronize();
    if( CUDA_ERROR_INVALID_CONTEXT != err) {
        fprintf(stderr, "Cuda error: %s in file '%s' in line %i\n",
                errorMessage, file, line );
        exit(-1);
    }
    err = cuCtxSynchronize();
    if( CUDA_SUCCESS != err) {
        fprintf(stderr, "Cuda error: %s in file '%s' in line %i\n",
                errorMessage, file, line );
        exit(-1);
    }
}

#ifndef STRCASECMP
#ifdef _WIN32
#define STRCASECMP  _stricmp
#else
#define STRCASECMP  strcasecmp
#endif
#endif

#ifndef STRNCASECMP
#ifdef _WIN32
#define STRNCASECMP _strnicmp
#else
#define STRNCASECMP strncasecmp
#endif
#endif

inline void __cutilDrvQAFinish(int argc, char **argv, bool bStatus)
{
    const char *sStatus[] = { "FAILED", "PASSED", "WAIVED", NULL };

    bool bFlag = false;
    for (int i=1; i < argc; i++) {
        if (!STRCASECMP(argv[i], "-qatest") || !STRCASECMP(argv[i], "-noprompt")) {
            bFlag |= true;
        }
    }

    if (bFlag) {
        printf("&&&& %s %s", sStatus[bStatus], argv[0]);
        for (int i=1; i < argc; i++) printf(" %s", argv[i]);
    } else {
        printf("[%s] test result\n%s\n", argv[0], sStatus[bStatus]);
    }
}

inline bool cutilDrvCudaDevCapabilities(int major_version, int minor_version, int deviceNum)
{
    int major, minor, dev;
    char device_name[256];

    cutilDrvSafeCallNoSync( cuDeviceGet(&dev, deviceNum) );
    cutilDrvSafeCallNoSync( cuDeviceComputeCapability(&major, &minor, dev));
    cutilDrvSafeCallNoSync( cuDeviceGetName(device_name, 256, dev) );

    if((major > major_version) ||
       (major == major_version && minor >= minor_version))
    {
        printf("> Device %d: < %s >, Compute SM %d.%d detected\n", dev, device_name, major, minor);
        return true;
    }
    else
    {
        printf("There is no device supporting CUDA compute capability %d.%d.\n", major_version, minor_version);
        return false;
    }
}

inline bool cutilDrvCudaCapabilities(int major_version, int minor_version)
{
    return cutilDrvCudaDevCapabilities(major_version, minor_version,0);
}

#endif // _CUTIL_INLINE_FUNCTIONS_DRVAPI_H_
