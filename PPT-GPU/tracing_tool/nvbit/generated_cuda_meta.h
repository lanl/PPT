/*
 * Copyright 2019 NVIDIA Corporation.  All rights reserved.
 *
 * NOTICE TO LICENSEE:
 *
 * This source code and/or documentation ("Licensed Deliverables") are
 * subject to NVIDIA intellectual property rights under U.S. and
 * international Copyright laws.
 *
 * These Licensed Deliverables contained herein is PROPRIETARY and
 * CONFIDENTIAL to NVIDIA and is being provided under the terms and
 * conditions of a form of NVIDIA software license agreement by and
 * between NVIDIA and Licensee ("License Agreement") or electronically
 * accepted by Licensee.  Notwithstanding any terms or conditions to
 * the contrary in the License Agreement, reproduction or disclosure
 * of the Licensed Deliverables to any third party without the express
 * written consent of NVIDIA is prohibited.
 *
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, NVIDIA MAKES NO REPRESENTATION ABOUT THE
 * SUITABILITY OF THESE LICENSED DELIVERABLES FOR ANY PURPOSE.  IT IS
 * PROVIDED "AS IS" WITHOUT EXPRESS OR IMPLIED WARRANTY OF ANY KIND.
 * NVIDIA DISCLAIMS ALL WARRANTIES WITH REGARD TO THESE LICENSED
 * DELIVERABLES, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY,
 * NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY
 * SPECIAL, INDIRECT, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, OR ANY
 * DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
 * WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS
 * ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE
 * OF THESE LICENSED DELIVERABLES.
 *
 * U.S. Government End Users.  These Licensed Deliverables are a
 * "commercial item" as that term is defined at 48 C.F.R. 2.101 (OCT
 * 1995), consisting of "commercial computer software" and "commercial
 * computer software documentation" as such terms are used in 48
 * C.F.R. 12.212 (SEPT 1995) and is provided to the U.S. Government
 * only as a commercial end item.  Consistent with 48 C.F.R.12.212 and
 * 48 C.F.R. 227.7202-1 through 227.7202-4 (JUNE 1995), all
 * U.S. Government End Users acquire the Licensed Deliverables with
 * only those rights set forth herein.
 *
 * Any use of the Licensed Deliverables in individual and commercial
 * software must include, in the user documentation and internal
 * comments to the code, the above Disclaimer and U.S. Government End
 * Users Notice.
 */

// This file is generated.  Any changes you make will be lost during the next clean build.
#ifndef __generated_cuda_meta_h__
#define __generated_cuda_meta_h__

// No dependent includes

// CUDA public interface, for type definitions and cu* function prototypes
#include "cuda.h"


// *************************************************************************
//      Definitions of structs to hold parameters for each function
// *************************************************************************

typedef struct cuGetErrorString_params_st {
    CUresult error;
    const char **pStr;
} cuGetErrorString_params;

typedef struct cuGetErrorName_params_st {
    CUresult error;
    const char **pStr;
} cuGetErrorName_params;

typedef struct cuInit_params_st {
    unsigned int Flags;
} cuInit_params;

typedef struct cuDriverGetVersion_params_st {
    int *driverVersion;
} cuDriverGetVersion_params;

typedef struct cuDeviceGet_params_st {
    CUdevice *device;
    int ordinal;
} cuDeviceGet_params;

typedef struct cuDeviceGetCount_params_st {
    int *count;
} cuDeviceGetCount_params;

typedef struct cuDeviceGetName_params_st {
    char *name;
    int len;
    CUdevice dev;
} cuDeviceGetName_params;

typedef struct cuDeviceGetUuid_params_st {
    CUuuid *uuid;
    CUdevice dev;
} cuDeviceGetUuid_params;

typedef struct cuDeviceTotalMem_v2_params_st {
    size_t *bytes;
    CUdevice dev;
} cuDeviceTotalMem_v2_params;

typedef struct cuDeviceGetAttribute_params_st {
    int *pi;
    CUdevice_attribute attrib;
    CUdevice dev;
} cuDeviceGetAttribute_params;

typedef struct cuDeviceGetNvSciSyncAttributes_params_st {
    void *nvSciSyncAttrList;
    CUdevice dev;
    int flags;
} cuDeviceGetNvSciSyncAttributes_params;

typedef struct cuDeviceGetProperties_params_st {
    CUdevprop *prop;
    CUdevice dev;
} cuDeviceGetProperties_params;

typedef struct cuDeviceComputeCapability_params_st {
    int *major;
    int *minor;
    CUdevice dev;
} cuDeviceComputeCapability_params;

typedef struct cuDevicePrimaryCtxRetain_params_st {
    CUcontext *pctx;
    CUdevice dev;
} cuDevicePrimaryCtxRetain_params;

typedef struct cuDevicePrimaryCtxRelease_v2_params_st {
    CUdevice dev;
} cuDevicePrimaryCtxRelease_v2_params;

typedef struct cuDevicePrimaryCtxSetFlags_v2_params_st {
    CUdevice dev;
    unsigned int flags;
} cuDevicePrimaryCtxSetFlags_v2_params;

typedef struct cuDevicePrimaryCtxGetState_params_st {
    CUdevice dev;
    unsigned int *flags;
    int *active;
} cuDevicePrimaryCtxGetState_params;

typedef struct cuDevicePrimaryCtxReset_v2_params_st {
    CUdevice dev;
} cuDevicePrimaryCtxReset_v2_params;

typedef struct cuCtxCreate_v2_params_st {
    CUcontext *pctx;
    unsigned int flags;
    CUdevice dev;
} cuCtxCreate_v2_params;

typedef struct cuCtxDestroy_v2_params_st {
    CUcontext ctx;
} cuCtxDestroy_v2_params;

typedef struct cuCtxPushCurrent_v2_params_st {
    CUcontext ctx;
} cuCtxPushCurrent_v2_params;

typedef struct cuCtxPopCurrent_v2_params_st {
    CUcontext *pctx;
} cuCtxPopCurrent_v2_params;

typedef struct cuCtxSetCurrent_params_st {
    CUcontext ctx;
} cuCtxSetCurrent_params;

typedef struct cuCtxGetCurrent_params_st {
    CUcontext *pctx;
} cuCtxGetCurrent_params;

typedef struct cuCtxGetDevice_params_st {
    CUdevice *device;
} cuCtxGetDevice_params;

typedef struct cuCtxGetFlags_params_st {
    unsigned int *flags;
} cuCtxGetFlags_params;

typedef struct cuCtxSetLimit_params_st {
    CUlimit limit;
    size_t value;
} cuCtxSetLimit_params;

typedef struct cuCtxGetLimit_params_st {
    size_t *pvalue;
    CUlimit limit;
} cuCtxGetLimit_params;

typedef struct cuCtxGetCacheConfig_params_st {
    CUfunc_cache *pconfig;
} cuCtxGetCacheConfig_params;

typedef struct cuCtxSetCacheConfig_params_st {
    CUfunc_cache config;
} cuCtxSetCacheConfig_params;

typedef struct cuCtxGetSharedMemConfig_params_st {
    CUsharedconfig *pConfig;
} cuCtxGetSharedMemConfig_params;

typedef struct cuCtxSetSharedMemConfig_params_st {
    CUsharedconfig config;
} cuCtxSetSharedMemConfig_params;

typedef struct cuCtxGetApiVersion_params_st {
    CUcontext ctx;
    unsigned int *version;
} cuCtxGetApiVersion_params;

typedef struct cuCtxGetStreamPriorityRange_params_st {
    int *leastPriority;
    int *greatestPriority;
} cuCtxGetStreamPriorityRange_params;

typedef struct cuCtxAttach_params_st {
    CUcontext *pctx;
    unsigned int flags;
} cuCtxAttach_params;

typedef struct cuCtxDetach_params_st {
    CUcontext ctx;
} cuCtxDetach_params;

typedef struct cuModuleLoad_params_st {
    CUmodule *module;
    const char *fname;
} cuModuleLoad_params;

typedef struct cuModuleLoadData_params_st {
    CUmodule *module;
    const void *image;
} cuModuleLoadData_params;

typedef struct cuModuleLoadDataEx_params_st {
    CUmodule *module;
    const void *image;
    unsigned int numOptions;
    CUjit_option *options;
    void **optionValues;
} cuModuleLoadDataEx_params;

typedef struct cuModuleLoadFatBinary_params_st {
    CUmodule *module;
    const void *fatCubin;
} cuModuleLoadFatBinary_params;

typedef struct cuModuleUnload_params_st {
    CUmodule hmod;
} cuModuleUnload_params;

typedef struct cuModuleGetFunction_params_st {
    CUfunction *hfunc;
    CUmodule hmod;
    const char *name;
} cuModuleGetFunction_params;

typedef struct cuModuleGetGlobal_v2_params_st {
    CUdeviceptr *dptr;
    size_t *bytes;
    CUmodule hmod;
    const char *name;
} cuModuleGetGlobal_v2_params;

typedef struct cuModuleGetTexRef_params_st {
    CUtexref *pTexRef;
    CUmodule hmod;
    const char *name;
} cuModuleGetTexRef_params;

typedef struct cuModuleGetSurfRef_params_st {
    CUsurfref *pSurfRef;
    CUmodule hmod;
    const char *name;
} cuModuleGetSurfRef_params;

typedef struct cuLinkCreate_v2_params_st {
    unsigned int numOptions;
    CUjit_option *options;
    void **optionValues;
    CUlinkState *stateOut;
} cuLinkCreate_v2_params;

typedef struct cuLinkAddData_v2_params_st {
    CUlinkState state;
    CUjitInputType type;
    void *data;
    size_t size;
    const char *name;
    unsigned int numOptions;
    CUjit_option *options;
    void **optionValues;
} cuLinkAddData_v2_params;

typedef struct cuLinkAddFile_v2_params_st {
    CUlinkState state;
    CUjitInputType type;
    const char *path;
    unsigned int numOptions;
    CUjit_option *options;
    void **optionValues;
} cuLinkAddFile_v2_params;

typedef struct cuLinkComplete_params_st {
    CUlinkState state;
    void **cubinOut;
    size_t *sizeOut;
} cuLinkComplete_params;

typedef struct cuLinkDestroy_params_st {
    CUlinkState state;
} cuLinkDestroy_params;

typedef struct cuMemGetInfo_v2_params_st {
    size_t *free;
    size_t *total;
} cuMemGetInfo_v2_params;

typedef struct cuMemAlloc_v2_params_st {
    CUdeviceptr *dptr;
    size_t bytesize;
} cuMemAlloc_v2_params;

typedef struct cuMemAllocPitch_v2_params_st {
    CUdeviceptr *dptr;
    size_t *pPitch;
    size_t WidthInBytes;
    size_t Height;
    unsigned int ElementSizeBytes;
} cuMemAllocPitch_v2_params;

typedef struct cuMemFree_v2_params_st {
    CUdeviceptr dptr;
} cuMemFree_v2_params;

typedef struct cuMemGetAddressRange_v2_params_st {
    CUdeviceptr *pbase;
    size_t *psize;
    CUdeviceptr dptr;
} cuMemGetAddressRange_v2_params;

typedef struct cuMemAllocHost_v2_params_st {
    void **pp;
    size_t bytesize;
} cuMemAllocHost_v2_params;

typedef struct cuMemFreeHost_params_st {
    void *p;
} cuMemFreeHost_params;

typedef struct cuMemHostAlloc_params_st {
    void **pp;
    size_t bytesize;
    unsigned int Flags;
} cuMemHostAlloc_params;

typedef struct cuMemHostGetDevicePointer_v2_params_st {
    CUdeviceptr *pdptr;
    void *p;
    unsigned int Flags;
} cuMemHostGetDevicePointer_v2_params;

typedef struct cuMemHostGetFlags_params_st {
    unsigned int *pFlags;
    void *p;
} cuMemHostGetFlags_params;

typedef struct cuMemAllocManaged_params_st {
    CUdeviceptr *dptr;
    size_t bytesize;
    unsigned int flags;
} cuMemAllocManaged_params;

typedef struct cuDeviceGetByPCIBusId_params_st {
    CUdevice *dev;
    const char *pciBusId;
} cuDeviceGetByPCIBusId_params;

typedef struct cuDeviceGetPCIBusId_params_st {
    char *pciBusId;
    int len;
    CUdevice dev;
} cuDeviceGetPCIBusId_params;

typedef struct cuIpcGetEventHandle_params_st {
    CUipcEventHandle *pHandle;
    CUevent event;
} cuIpcGetEventHandle_params;

typedef struct cuIpcOpenEventHandle_params_st {
    CUevent *phEvent;
    CUipcEventHandle handle;
} cuIpcOpenEventHandle_params;

typedef struct cuIpcGetMemHandle_params_st {
    CUipcMemHandle *pHandle;
    CUdeviceptr dptr;
} cuIpcGetMemHandle_params;

typedef struct cuIpcOpenMemHandle_v2_params_st {
    CUdeviceptr *pdptr;
    CUipcMemHandle handle;
    unsigned int Flags;
} cuIpcOpenMemHandle_v2_params;

typedef struct cuIpcCloseMemHandle_params_st {
    CUdeviceptr dptr;
} cuIpcCloseMemHandle_params;

typedef struct cuMemHostRegister_v2_params_st {
    void *p;
    size_t bytesize;
    unsigned int Flags;
} cuMemHostRegister_v2_params;

typedef struct cuMemHostUnregister_params_st {
    void *p;
} cuMemHostUnregister_params;

typedef struct cuMemcpy_ptds_params_st {
    CUdeviceptr dst;
    CUdeviceptr src;
    size_t ByteCount;
} cuMemcpy_ptds_params;

typedef struct cuMemcpyPeer_ptds_params_st {
    CUdeviceptr dstDevice;
    CUcontext dstContext;
    CUdeviceptr srcDevice;
    CUcontext srcContext;
    size_t ByteCount;
} cuMemcpyPeer_ptds_params;

typedef struct cuMemcpyHtoD_v2_ptds_params_st {
    CUdeviceptr dstDevice;
    const void *srcHost;
    size_t ByteCount;
} cuMemcpyHtoD_v2_ptds_params;

typedef struct cuMemcpyDtoH_v2_ptds_params_st {
    void *dstHost;
    CUdeviceptr srcDevice;
    size_t ByteCount;
} cuMemcpyDtoH_v2_ptds_params;

typedef struct cuMemcpyDtoD_v2_ptds_params_st {
    CUdeviceptr dstDevice;
    CUdeviceptr srcDevice;
    size_t ByteCount;
} cuMemcpyDtoD_v2_ptds_params;

typedef struct cuMemcpyDtoA_v2_ptds_params_st {
    CUarray dstArray;
    size_t dstOffset;
    CUdeviceptr srcDevice;
    size_t ByteCount;
} cuMemcpyDtoA_v2_ptds_params;

typedef struct cuMemcpyAtoD_v2_ptds_params_st {
    CUdeviceptr dstDevice;
    CUarray srcArray;
    size_t srcOffset;
    size_t ByteCount;
} cuMemcpyAtoD_v2_ptds_params;

typedef struct cuMemcpyHtoA_v2_ptds_params_st {
    CUarray dstArray;
    size_t dstOffset;
    const void *srcHost;
    size_t ByteCount;
} cuMemcpyHtoA_v2_ptds_params;

typedef struct cuMemcpyAtoH_v2_ptds_params_st {
    void *dstHost;
    CUarray srcArray;
    size_t srcOffset;
    size_t ByteCount;
} cuMemcpyAtoH_v2_ptds_params;

typedef struct cuMemcpyAtoA_v2_ptds_params_st {
    CUarray dstArray;
    size_t dstOffset;
    CUarray srcArray;
    size_t srcOffset;
    size_t ByteCount;
} cuMemcpyAtoA_v2_ptds_params;

typedef struct cuMemcpy2D_v2_ptds_params_st {
    const CUDA_MEMCPY2D *pCopy;
} cuMemcpy2D_v2_ptds_params;

typedef struct cuMemcpy2DUnaligned_v2_ptds_params_st {
    const CUDA_MEMCPY2D *pCopy;
} cuMemcpy2DUnaligned_v2_ptds_params;

typedef struct cuMemcpy3D_v2_ptds_params_st {
    const CUDA_MEMCPY3D *pCopy;
} cuMemcpy3D_v2_ptds_params;

typedef struct cuMemcpy3DPeer_ptds_params_st {
    const CUDA_MEMCPY3D_PEER *pCopy;
} cuMemcpy3DPeer_ptds_params;

typedef struct cuMemcpyAsync_ptsz_params_st {
    CUdeviceptr dst;
    CUdeviceptr src;
    size_t ByteCount;
    CUstream hStream;
} cuMemcpyAsync_ptsz_params;

typedef struct cuMemcpyPeerAsync_ptsz_params_st {
    CUdeviceptr dstDevice;
    CUcontext dstContext;
    CUdeviceptr srcDevice;
    CUcontext srcContext;
    size_t ByteCount;
    CUstream hStream;
} cuMemcpyPeerAsync_ptsz_params;

typedef struct cuMemcpyHtoDAsync_v2_ptsz_params_st {
    CUdeviceptr dstDevice;
    const void *srcHost;
    size_t ByteCount;
    CUstream hStream;
} cuMemcpyHtoDAsync_v2_ptsz_params;

typedef struct cuMemcpyDtoHAsync_v2_ptsz_params_st {
    void *dstHost;
    CUdeviceptr srcDevice;
    size_t ByteCount;
    CUstream hStream;
} cuMemcpyDtoHAsync_v2_ptsz_params;

typedef struct cuMemcpyDtoDAsync_v2_ptsz_params_st {
    CUdeviceptr dstDevice;
    CUdeviceptr srcDevice;
    size_t ByteCount;
    CUstream hStream;
} cuMemcpyDtoDAsync_v2_ptsz_params;

typedef struct cuMemcpyHtoAAsync_v2_ptsz_params_st {
    CUarray dstArray;
    size_t dstOffset;
    const void *srcHost;
    size_t ByteCount;
    CUstream hStream;
} cuMemcpyHtoAAsync_v2_ptsz_params;

typedef struct cuMemcpyAtoHAsync_v2_ptsz_params_st {
    void *dstHost;
    CUarray srcArray;
    size_t srcOffset;
    size_t ByteCount;
    CUstream hStream;
} cuMemcpyAtoHAsync_v2_ptsz_params;

typedef struct cuMemcpy2DAsync_v2_ptsz_params_st {
    const CUDA_MEMCPY2D *pCopy;
    CUstream hStream;
} cuMemcpy2DAsync_v2_ptsz_params;

typedef struct cuMemcpy3DAsync_v2_ptsz_params_st {
    const CUDA_MEMCPY3D *pCopy;
    CUstream hStream;
} cuMemcpy3DAsync_v2_ptsz_params;

typedef struct cuMemcpy3DPeerAsync_ptsz_params_st {
    const CUDA_MEMCPY3D_PEER *pCopy;
    CUstream hStream;
} cuMemcpy3DPeerAsync_ptsz_params;

typedef struct cuMemsetD8_v2_ptds_params_st {
    CUdeviceptr dstDevice;
    unsigned char uc;
    size_t N;
} cuMemsetD8_v2_ptds_params;

typedef struct cuMemsetD16_v2_ptds_params_st {
    CUdeviceptr dstDevice;
    unsigned short us;
    size_t N;
} cuMemsetD16_v2_ptds_params;

typedef struct cuMemsetD32_v2_ptds_params_st {
    CUdeviceptr dstDevice;
    unsigned int ui;
    size_t N;
} cuMemsetD32_v2_ptds_params;

typedef struct cuMemsetD2D8_v2_ptds_params_st {
    CUdeviceptr dstDevice;
    size_t dstPitch;
    unsigned char uc;
    size_t Width;
    size_t Height;
} cuMemsetD2D8_v2_ptds_params;

typedef struct cuMemsetD2D16_v2_ptds_params_st {
    CUdeviceptr dstDevice;
    size_t dstPitch;
    unsigned short us;
    size_t Width;
    size_t Height;
} cuMemsetD2D16_v2_ptds_params;

typedef struct cuMemsetD2D32_v2_ptds_params_st {
    CUdeviceptr dstDevice;
    size_t dstPitch;
    unsigned int ui;
    size_t Width;
    size_t Height;
} cuMemsetD2D32_v2_ptds_params;

typedef struct cuMemsetD8Async_ptsz_params_st {
    CUdeviceptr dstDevice;
    unsigned char uc;
    size_t N;
    CUstream hStream;
} cuMemsetD8Async_ptsz_params;

typedef struct cuMemsetD16Async_ptsz_params_st {
    CUdeviceptr dstDevice;
    unsigned short us;
    size_t N;
    CUstream hStream;
} cuMemsetD16Async_ptsz_params;

typedef struct cuMemsetD32Async_ptsz_params_st {
    CUdeviceptr dstDevice;
    unsigned int ui;
    size_t N;
    CUstream hStream;
} cuMemsetD32Async_ptsz_params;

typedef struct cuMemsetD2D8Async_ptsz_params_st {
    CUdeviceptr dstDevice;
    size_t dstPitch;
    unsigned char uc;
    size_t Width;
    size_t Height;
    CUstream hStream;
} cuMemsetD2D8Async_ptsz_params;

typedef struct cuMemsetD2D16Async_ptsz_params_st {
    CUdeviceptr dstDevice;
    size_t dstPitch;
    unsigned short us;
    size_t Width;
    size_t Height;
    CUstream hStream;
} cuMemsetD2D16Async_ptsz_params;

typedef struct cuMemsetD2D32Async_ptsz_params_st {
    CUdeviceptr dstDevice;
    size_t dstPitch;
    unsigned int ui;
    size_t Width;
    size_t Height;
    CUstream hStream;
} cuMemsetD2D32Async_ptsz_params;

typedef struct cuArrayCreate_v2_params_st {
    CUarray *pHandle;
    const CUDA_ARRAY_DESCRIPTOR *pAllocateArray;
} cuArrayCreate_v2_params;

typedef struct cuArrayGetDescriptor_v2_params_st {
    CUDA_ARRAY_DESCRIPTOR *pArrayDescriptor;
    CUarray hArray;
} cuArrayGetDescriptor_v2_params;

typedef struct cuArrayDestroy_params_st {
    CUarray hArray;
} cuArrayDestroy_params;

typedef struct cuArray3DCreate_v2_params_st {
    CUarray *pHandle;
    const CUDA_ARRAY3D_DESCRIPTOR *pAllocateArray;
} cuArray3DCreate_v2_params;

typedef struct cuArray3DGetDescriptor_v2_params_st {
    CUDA_ARRAY3D_DESCRIPTOR *pArrayDescriptor;
    CUarray hArray;
} cuArray3DGetDescriptor_v2_params;

typedef struct cuMipmappedArrayCreate_params_st {
    CUmipmappedArray *pHandle;
    const CUDA_ARRAY3D_DESCRIPTOR *pMipmappedArrayDesc;
    unsigned int numMipmapLevels;
} cuMipmappedArrayCreate_params;

typedef struct cuMipmappedArrayGetLevel_params_st {
    CUarray *pLevelArray;
    CUmipmappedArray hMipmappedArray;
    unsigned int level;
} cuMipmappedArrayGetLevel_params;

typedef struct cuMipmappedArrayDestroy_params_st {
    CUmipmappedArray hMipmappedArray;
} cuMipmappedArrayDestroy_params;

typedef struct cuMemAddressReserve_params_st {
    CUdeviceptr *ptr;
    size_t size;
    size_t alignment;
    CUdeviceptr addr;
    unsigned long long flags;
} cuMemAddressReserve_params;

typedef struct cuMemAddressFree_params_st {
    CUdeviceptr ptr;
    size_t size;
} cuMemAddressFree_params;

typedef struct cuMemCreate_params_st {
    CUmemGenericAllocationHandle *handle;
    size_t size;
    const CUmemAllocationProp *prop;
    unsigned long long flags;
} cuMemCreate_params;

typedef struct cuMemRelease_params_st {
    CUmemGenericAllocationHandle handle;
} cuMemRelease_params;

typedef struct cuMemMap_params_st {
    CUdeviceptr ptr;
    size_t size;
    size_t offset;
    CUmemGenericAllocationHandle handle;
    unsigned long long flags;
} cuMemMap_params;

typedef struct cuMemUnmap_params_st {
    CUdeviceptr ptr;
    size_t size;
} cuMemUnmap_params;

typedef struct cuMemSetAccess_params_st {
    CUdeviceptr ptr;
    size_t size;
    const CUmemAccessDesc *desc;
    size_t count;
} cuMemSetAccess_params;

typedef struct cuMemGetAccess_params_st {
    unsigned long long *flags;
    const CUmemLocation *location;
    CUdeviceptr ptr;
} cuMemGetAccess_params;

typedef struct cuMemExportToShareableHandle_params_st {
    void *shareableHandle;
    CUmemGenericAllocationHandle handle;
    CUmemAllocationHandleType handleType;
    unsigned long long flags;
} cuMemExportToShareableHandle_params;

typedef struct cuMemImportFromShareableHandle_params_st {
    CUmemGenericAllocationHandle *handle;
    void *osHandle;
    CUmemAllocationHandleType shHandleType;
} cuMemImportFromShareableHandle_params;

typedef struct cuMemGetAllocationGranularity_params_st {
    size_t *granularity;
    const CUmemAllocationProp *prop;
    CUmemAllocationGranularity_flags option;
} cuMemGetAllocationGranularity_params;

typedef struct cuMemGetAllocationPropertiesFromHandle_params_st {
    CUmemAllocationProp *prop;
    CUmemGenericAllocationHandle handle;
} cuMemGetAllocationPropertiesFromHandle_params;

typedef struct cuMemRetainAllocationHandle_params_st {
    CUmemGenericAllocationHandle *handle;
    void *addr;
} cuMemRetainAllocationHandle_params;

typedef struct cuPointerGetAttribute_params_st {
    void *data;
    CUpointer_attribute attribute;
    CUdeviceptr ptr;
} cuPointerGetAttribute_params;

typedef struct cuMemPrefetchAsync_ptsz_params_st {
    CUdeviceptr devPtr;
    size_t count;
    CUdevice dstDevice;
    CUstream hStream;
} cuMemPrefetchAsync_ptsz_params;

typedef struct cuMemAdvise_params_st {
    CUdeviceptr devPtr;
    size_t count;
    CUmem_advise advice;
    CUdevice device;
} cuMemAdvise_params;

typedef struct cuMemRangeGetAttribute_params_st {
    void *data;
    size_t dataSize;
    CUmem_range_attribute attribute;
    CUdeviceptr devPtr;
    size_t count;
} cuMemRangeGetAttribute_params;

typedef struct cuMemRangeGetAttributes_params_st {
    void **data;
    size_t *dataSizes;
    CUmem_range_attribute *attributes;
    size_t numAttributes;
    CUdeviceptr devPtr;
    size_t count;
} cuMemRangeGetAttributes_params;

typedef struct cuPointerSetAttribute_params_st {
    const void *value;
    CUpointer_attribute attribute;
    CUdeviceptr ptr;
} cuPointerSetAttribute_params;

typedef struct cuPointerGetAttributes_params_st {
    unsigned int numAttributes;
    CUpointer_attribute *attributes;
    void **data;
    CUdeviceptr ptr;
} cuPointerGetAttributes_params;

typedef struct cuStreamCreate_params_st {
    CUstream *phStream;
    unsigned int Flags;
} cuStreamCreate_params;

typedef struct cuStreamCreateWithPriority_params_st {
    CUstream *phStream;
    unsigned int flags;
    int priority;
} cuStreamCreateWithPriority_params;

typedef struct cuStreamGetPriority_ptsz_params_st {
    CUstream hStream;
    int *priority;
} cuStreamGetPriority_ptsz_params;

typedef struct cuStreamGetFlags_ptsz_params_st {
    CUstream hStream;
    unsigned int *flags;
} cuStreamGetFlags_ptsz_params;

typedef struct cuStreamGetCtx_ptsz_params_st {
    CUstream hStream;
    CUcontext *pctx;
} cuStreamGetCtx_ptsz_params;

typedef struct cuStreamWaitEvent_ptsz_params_st {
    CUstream hStream;
    CUevent hEvent;
    unsigned int Flags;
} cuStreamWaitEvent_ptsz_params;

typedef struct cuStreamAddCallback_ptsz_params_st {
    CUstream hStream;
    CUstreamCallback callback;
    void *userData;
    unsigned int flags;
} cuStreamAddCallback_ptsz_params;

typedef struct cuStreamBeginCapture_v2_ptsz_params_st {
    CUstream hStream;
    CUstreamCaptureMode mode;
} cuStreamBeginCapture_v2_ptsz_params;

typedef struct cuThreadExchangeStreamCaptureMode_params_st {
    CUstreamCaptureMode *mode;
} cuThreadExchangeStreamCaptureMode_params;

typedef struct cuStreamEndCapture_ptsz_params_st {
    CUstream hStream;
    CUgraph *phGraph;
} cuStreamEndCapture_ptsz_params;

typedef struct cuStreamIsCapturing_ptsz_params_st {
    CUstream hStream;
    CUstreamCaptureStatus *captureStatus;
} cuStreamIsCapturing_ptsz_params;

typedef struct cuStreamGetCaptureInfo_ptsz_params_st {
    CUstream hStream;
    CUstreamCaptureStatus *captureStatus;
    cuuint64_t *id;
} cuStreamGetCaptureInfo_ptsz_params;

typedef struct cuStreamAttachMemAsync_ptsz_params_st {
    CUstream hStream;
    CUdeviceptr dptr;
    size_t length;
    unsigned int flags;
} cuStreamAttachMemAsync_ptsz_params;

typedef struct cuStreamQuery_ptsz_params_st {
    CUstream hStream;
} cuStreamQuery_ptsz_params;

typedef struct cuStreamSynchronize_ptsz_params_st {
    CUstream hStream;
} cuStreamSynchronize_ptsz_params;

typedef struct cuStreamDestroy_v2_params_st {
    CUstream hStream;
} cuStreamDestroy_v2_params;

typedef struct cuStreamCopyAttributes_ptsz_params_st {
    CUstream dst;
    CUstream src;
} cuStreamCopyAttributes_ptsz_params;

typedef struct cuStreamGetAttribute_ptsz_params_st {
    CUstream hStream;
    CUstreamAttrID attr;
    CUstreamAttrValue *value_out;
} cuStreamGetAttribute_ptsz_params;

typedef struct cuStreamSetAttribute_ptsz_params_st {
    CUstream hStream;
    CUstreamAttrID attr;
    const CUstreamAttrValue *value;
} cuStreamSetAttribute_ptsz_params;

typedef struct cuEventCreate_params_st {
    CUevent *phEvent;
    unsigned int Flags;
} cuEventCreate_params;

typedef struct cuEventRecord_ptsz_params_st {
    CUevent hEvent;
    CUstream hStream;
} cuEventRecord_ptsz_params;

typedef struct cuEventQuery_params_st {
    CUevent hEvent;
} cuEventQuery_params;

typedef struct cuEventSynchronize_params_st {
    CUevent hEvent;
} cuEventSynchronize_params;

typedef struct cuEventDestroy_v2_params_st {
    CUevent hEvent;
} cuEventDestroy_v2_params;

typedef struct cuEventElapsedTime_params_st {
    float *pMilliseconds;
    CUevent hStart;
    CUevent hEnd;
} cuEventElapsedTime_params;

typedef struct cuImportExternalMemory_params_st {
    CUexternalMemory *extMem_out;
    const CUDA_EXTERNAL_MEMORY_HANDLE_DESC *memHandleDesc;
} cuImportExternalMemory_params;

typedef struct cuExternalMemoryGetMappedBuffer_params_st {
    CUdeviceptr *devPtr;
    CUexternalMemory extMem;
    const CUDA_EXTERNAL_MEMORY_BUFFER_DESC *bufferDesc;
} cuExternalMemoryGetMappedBuffer_params;

typedef struct cuExternalMemoryGetMappedMipmappedArray_params_st {
    CUmipmappedArray *mipmap;
    CUexternalMemory extMem;
    const CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC *mipmapDesc;
} cuExternalMemoryGetMappedMipmappedArray_params;

typedef struct cuDestroyExternalMemory_params_st {
    CUexternalMemory extMem;
} cuDestroyExternalMemory_params;

typedef struct cuImportExternalSemaphore_params_st {
    CUexternalSemaphore *extSem_out;
    const CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC *semHandleDesc;
} cuImportExternalSemaphore_params;

typedef struct cuSignalExternalSemaphoresAsync_ptsz_params_st {
    const CUexternalSemaphore *extSemArray;
    const CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS *paramsArray;
    unsigned int numExtSems;
    CUstream stream;
} cuSignalExternalSemaphoresAsync_ptsz_params;

typedef struct cuWaitExternalSemaphoresAsync_ptsz_params_st {
    const CUexternalSemaphore *extSemArray;
    const CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS *paramsArray;
    unsigned int numExtSems;
    CUstream stream;
} cuWaitExternalSemaphoresAsync_ptsz_params;

typedef struct cuDestroyExternalSemaphore_params_st {
    CUexternalSemaphore extSem;
} cuDestroyExternalSemaphore_params;

typedef struct cuStreamWaitValue32_ptsz_params_st {
    CUstream stream;
    CUdeviceptr addr;
    cuuint32_t value;
    unsigned int flags;
} cuStreamWaitValue32_ptsz_params;

typedef struct cuStreamWaitValue64_ptsz_params_st {
    CUstream stream;
    CUdeviceptr addr;
    cuuint64_t value;
    unsigned int flags;
} cuStreamWaitValue64_ptsz_params;

typedef struct cuStreamWriteValue32_ptsz_params_st {
    CUstream stream;
    CUdeviceptr addr;
    cuuint32_t value;
    unsigned int flags;
} cuStreamWriteValue32_ptsz_params;

typedef struct cuStreamWriteValue64_ptsz_params_st {
    CUstream stream;
    CUdeviceptr addr;
    cuuint64_t value;
    unsigned int flags;
} cuStreamWriteValue64_ptsz_params;

typedef struct cuStreamBatchMemOp_ptsz_params_st {
    CUstream stream;
    unsigned int count;
    CUstreamBatchMemOpParams *paramArray;
    unsigned int flags;
} cuStreamBatchMemOp_ptsz_params;

typedef struct cuFuncGetAttribute_params_st {
    int *pi;
    CUfunction_attribute attrib;
    CUfunction hfunc;
} cuFuncGetAttribute_params;

typedef struct cuFuncSetAttribute_params_st {
    CUfunction hfunc;
    CUfunction_attribute attrib;
    int value;
} cuFuncSetAttribute_params;

typedef struct cuFuncSetCacheConfig_params_st {
    CUfunction hfunc;
    CUfunc_cache config;
} cuFuncSetCacheConfig_params;

typedef struct cuFuncSetSharedMemConfig_params_st {
    CUfunction hfunc;
    CUsharedconfig config;
} cuFuncSetSharedMemConfig_params;

typedef struct cuLaunchKernel_ptsz_params_st {
    CUfunction f;
    unsigned int gridDimX;
    unsigned int gridDimY;
    unsigned int gridDimZ;
    unsigned int blockDimX;
    unsigned int blockDimY;
    unsigned int blockDimZ;
    unsigned int sharedMemBytes;
    CUstream hStream;
    void **kernelParams;
    void **extra;
} cuLaunchKernel_ptsz_params;

typedef struct cuLaunchCooperativeKernel_ptsz_params_st {
    CUfunction f;
    unsigned int gridDimX;
    unsigned int gridDimY;
    unsigned int gridDimZ;
    unsigned int blockDimX;
    unsigned int blockDimY;
    unsigned int blockDimZ;
    unsigned int sharedMemBytes;
    CUstream hStream;
    void **kernelParams;
} cuLaunchCooperativeKernel_ptsz_params;

typedef struct cuLaunchCooperativeKernelMultiDevice_params_st {
    CUDA_LAUNCH_PARAMS *launchParamsList;
    unsigned int numDevices;
    unsigned int flags;
} cuLaunchCooperativeKernelMultiDevice_params;

typedef struct cuLaunchHostFunc_ptsz_params_st {
    CUstream hStream;
    CUhostFn fn;
    void *userData;
} cuLaunchHostFunc_ptsz_params;

typedef struct cuFuncSetBlockShape_params_st {
    CUfunction hfunc;
    int x;
    int y;
    int z;
} cuFuncSetBlockShape_params;

typedef struct cuFuncSetSharedSize_params_st {
    CUfunction hfunc;
    unsigned int bytes;
} cuFuncSetSharedSize_params;

typedef struct cuParamSetSize_params_st {
    CUfunction hfunc;
    unsigned int numbytes;
} cuParamSetSize_params;

typedef struct cuParamSeti_params_st {
    CUfunction hfunc;
    int offset;
    unsigned int value;
} cuParamSeti_params;

typedef struct cuParamSetf_params_st {
    CUfunction hfunc;
    int offset;
    float value;
} cuParamSetf_params;

typedef struct cuParamSetv_params_st {
    CUfunction hfunc;
    int offset;
    void *ptr;
    unsigned int numbytes;
} cuParamSetv_params;

typedef struct cuLaunch_params_st {
    CUfunction f;
} cuLaunch_params;

typedef struct cuLaunchGrid_params_st {
    CUfunction f;
    int grid_width;
    int grid_height;
} cuLaunchGrid_params;

typedef struct cuLaunchGridAsync_params_st {
    CUfunction f;
    int grid_width;
    int grid_height;
    CUstream hStream;
} cuLaunchGridAsync_params;

typedef struct cuParamSetTexRef_params_st {
    CUfunction hfunc;
    int texunit;
    CUtexref hTexRef;
} cuParamSetTexRef_params;

typedef struct cuGraphCreate_params_st {
    CUgraph *phGraph;
    unsigned int flags;
} cuGraphCreate_params;

typedef struct cuGraphAddKernelNode_params_st {
    CUgraphNode *phGraphNode;
    CUgraph hGraph;
    const CUgraphNode *dependencies;
    size_t numDependencies;
    const CUDA_KERNEL_NODE_PARAMS *nodeParams;
} cuGraphAddKernelNode_params;

typedef struct cuGraphKernelNodeGetParams_params_st {
    CUgraphNode hNode;
    CUDA_KERNEL_NODE_PARAMS *nodeParams;
} cuGraphKernelNodeGetParams_params;

typedef struct cuGraphKernelNodeSetParams_params_st {
    CUgraphNode hNode;
    const CUDA_KERNEL_NODE_PARAMS *nodeParams;
} cuGraphKernelNodeSetParams_params;

typedef struct cuGraphAddMemcpyNode_params_st {
    CUgraphNode *phGraphNode;
    CUgraph hGraph;
    const CUgraphNode *dependencies;
    size_t numDependencies;
    const CUDA_MEMCPY3D *copyParams;
    CUcontext ctx;
} cuGraphAddMemcpyNode_params;

typedef struct cuGraphMemcpyNodeGetParams_params_st {
    CUgraphNode hNode;
    CUDA_MEMCPY3D *nodeParams;
} cuGraphMemcpyNodeGetParams_params;

typedef struct cuGraphMemcpyNodeSetParams_params_st {
    CUgraphNode hNode;
    const CUDA_MEMCPY3D *nodeParams;
} cuGraphMemcpyNodeSetParams_params;

typedef struct cuGraphAddMemsetNode_params_st {
    CUgraphNode *phGraphNode;
    CUgraph hGraph;
    const CUgraphNode *dependencies;
    size_t numDependencies;
    const CUDA_MEMSET_NODE_PARAMS *memsetParams;
    CUcontext ctx;
} cuGraphAddMemsetNode_params;

typedef struct cuGraphMemsetNodeGetParams_params_st {
    CUgraphNode hNode;
    CUDA_MEMSET_NODE_PARAMS *nodeParams;
} cuGraphMemsetNodeGetParams_params;

typedef struct cuGraphMemsetNodeSetParams_params_st {
    CUgraphNode hNode;
    const CUDA_MEMSET_NODE_PARAMS *nodeParams;
} cuGraphMemsetNodeSetParams_params;

typedef struct cuGraphAddHostNode_params_st {
    CUgraphNode *phGraphNode;
    CUgraph hGraph;
    const CUgraphNode *dependencies;
    size_t numDependencies;
    const CUDA_HOST_NODE_PARAMS *nodeParams;
} cuGraphAddHostNode_params;

typedef struct cuGraphHostNodeGetParams_params_st {
    CUgraphNode hNode;
    CUDA_HOST_NODE_PARAMS *nodeParams;
} cuGraphHostNodeGetParams_params;

typedef struct cuGraphHostNodeSetParams_params_st {
    CUgraphNode hNode;
    const CUDA_HOST_NODE_PARAMS *nodeParams;
} cuGraphHostNodeSetParams_params;

typedef struct cuGraphAddChildGraphNode_params_st {
    CUgraphNode *phGraphNode;
    CUgraph hGraph;
    const CUgraphNode *dependencies;
    size_t numDependencies;
    CUgraph childGraph;
} cuGraphAddChildGraphNode_params;

typedef struct cuGraphChildGraphNodeGetGraph_params_st {
    CUgraphNode hNode;
    CUgraph *phGraph;
} cuGraphChildGraphNodeGetGraph_params;

typedef struct cuGraphAddEmptyNode_params_st {
    CUgraphNode *phGraphNode;
    CUgraph hGraph;
    const CUgraphNode *dependencies;
    size_t numDependencies;
} cuGraphAddEmptyNode_params;

typedef struct cuGraphClone_params_st {
    CUgraph *phGraphClone;
    CUgraph originalGraph;
} cuGraphClone_params;

typedef struct cuGraphNodeFindInClone_params_st {
    CUgraphNode *phNode;
    CUgraphNode hOriginalNode;
    CUgraph hClonedGraph;
} cuGraphNodeFindInClone_params;

typedef struct cuGraphNodeGetType_params_st {
    CUgraphNode hNode;
    CUgraphNodeType *type;
} cuGraphNodeGetType_params;

typedef struct cuGraphGetNodes_params_st {
    CUgraph hGraph;
    CUgraphNode *nodes;
    size_t *numNodes;
} cuGraphGetNodes_params;

typedef struct cuGraphGetRootNodes_params_st {
    CUgraph hGraph;
    CUgraphNode *rootNodes;
    size_t *numRootNodes;
} cuGraphGetRootNodes_params;

typedef struct cuGraphGetEdges_params_st {
    CUgraph hGraph;
    CUgraphNode *from;
    CUgraphNode *to;
    size_t *numEdges;
} cuGraphGetEdges_params;

typedef struct cuGraphNodeGetDependencies_params_st {
    CUgraphNode hNode;
    CUgraphNode *dependencies;
    size_t *numDependencies;
} cuGraphNodeGetDependencies_params;

typedef struct cuGraphNodeGetDependentNodes_params_st {
    CUgraphNode hNode;
    CUgraphNode *dependentNodes;
    size_t *numDependentNodes;
} cuGraphNodeGetDependentNodes_params;

typedef struct cuGraphAddDependencies_params_st {
    CUgraph hGraph;
    const CUgraphNode *from;
    const CUgraphNode *to;
    size_t numDependencies;
} cuGraphAddDependencies_params;

typedef struct cuGraphRemoveDependencies_params_st {
    CUgraph hGraph;
    const CUgraphNode *from;
    const CUgraphNode *to;
    size_t numDependencies;
} cuGraphRemoveDependencies_params;

typedef struct cuGraphDestroyNode_params_st {
    CUgraphNode hNode;
} cuGraphDestroyNode_params;

typedef struct cuGraphInstantiate_v2_params_st {
    CUgraphExec *phGraphExec;
    CUgraph hGraph;
    CUgraphNode *phErrorNode;
    char *logBuffer;
    size_t bufferSize;
} cuGraphInstantiate_v2_params;

typedef struct cuGraphExecKernelNodeSetParams_params_st {
    CUgraphExec hGraphExec;
    CUgraphNode hNode;
    const CUDA_KERNEL_NODE_PARAMS *nodeParams;
} cuGraphExecKernelNodeSetParams_params;

typedef struct cuGraphExecMemcpyNodeSetParams_params_st {
    CUgraphExec hGraphExec;
    CUgraphNode hNode;
    const CUDA_MEMCPY3D *copyParams;
    CUcontext ctx;
} cuGraphExecMemcpyNodeSetParams_params;

typedef struct cuGraphExecMemsetNodeSetParams_params_st {
    CUgraphExec hGraphExec;
    CUgraphNode hNode;
    const CUDA_MEMSET_NODE_PARAMS *memsetParams;
    CUcontext ctx;
} cuGraphExecMemsetNodeSetParams_params;

typedef struct cuGraphExecHostNodeSetParams_params_st {
    CUgraphExec hGraphExec;
    CUgraphNode hNode;
    const CUDA_HOST_NODE_PARAMS *nodeParams;
} cuGraphExecHostNodeSetParams_params;

typedef struct cuGraphLaunch_ptsz_params_st {
    CUgraphExec hGraphExec;
    CUstream hStream;
} cuGraphLaunch_ptsz_params;

typedef struct cuGraphExecDestroy_params_st {
    CUgraphExec hGraphExec;
} cuGraphExecDestroy_params;

typedef struct cuGraphDestroy_params_st {
    CUgraph hGraph;
} cuGraphDestroy_params;

typedef struct cuGraphExecUpdate_params_st {
    CUgraphExec hGraphExec;
    CUgraph hGraph;
    CUgraphNode *hErrorNode_out;
    CUgraphExecUpdateResult *updateResult_out;
} cuGraphExecUpdate_params;

typedef struct cuGraphKernelNodeCopyAttributes_params_st {
    CUgraphNode dst;
    CUgraphNode src;
} cuGraphKernelNodeCopyAttributes_params;

typedef struct cuGraphKernelNodeGetAttribute_params_st {
    CUgraphNode hNode;
    CUkernelNodeAttrID attr;
    CUkernelNodeAttrValue *value_out;
} cuGraphKernelNodeGetAttribute_params;

typedef struct cuGraphKernelNodeSetAttribute_params_st {
    CUgraphNode hNode;
    CUkernelNodeAttrID attr;
    const CUkernelNodeAttrValue *value;
} cuGraphKernelNodeSetAttribute_params;

typedef struct cuOccupancyMaxActiveBlocksPerMultiprocessor_params_st {
    int *numBlocks;
    CUfunction func;
    int blockSize;
    size_t dynamicSMemSize;
} cuOccupancyMaxActiveBlocksPerMultiprocessor_params;

typedef struct cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags_params_st {
    int *numBlocks;
    CUfunction func;
    int blockSize;
    size_t dynamicSMemSize;
    unsigned int flags;
} cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags_params;

typedef struct cuOccupancyMaxPotentialBlockSize_params_st {
    int *minGridSize;
    int *blockSize;
    CUfunction func;
    CUoccupancyB2DSize blockSizeToDynamicSMemSize;
    size_t dynamicSMemSize;
    int blockSizeLimit;
} cuOccupancyMaxPotentialBlockSize_params;

typedef struct cuOccupancyMaxPotentialBlockSizeWithFlags_params_st {
    int *minGridSize;
    int *blockSize;
    CUfunction func;
    CUoccupancyB2DSize blockSizeToDynamicSMemSize;
    size_t dynamicSMemSize;
    int blockSizeLimit;
    unsigned int flags;
} cuOccupancyMaxPotentialBlockSizeWithFlags_params;

typedef struct cuOccupancyAvailableDynamicSMemPerBlock_params_st {
    size_t *dynamicSmemSize;
    CUfunction func;
    int numBlocks;
    int blockSize;
} cuOccupancyAvailableDynamicSMemPerBlock_params;

typedef struct cuTexRefSetArray_params_st {
    CUtexref hTexRef;
    CUarray hArray;
    unsigned int Flags;
} cuTexRefSetArray_params;

typedef struct cuTexRefSetMipmappedArray_params_st {
    CUtexref hTexRef;
    CUmipmappedArray hMipmappedArray;
    unsigned int Flags;
} cuTexRefSetMipmappedArray_params;

typedef struct cuTexRefSetAddress_v2_params_st {
    size_t *ByteOffset;
    CUtexref hTexRef;
    CUdeviceptr dptr;
    size_t bytes;
} cuTexRefSetAddress_v2_params;

typedef struct cuTexRefSetAddress2D_v3_params_st {
    CUtexref hTexRef;
    const CUDA_ARRAY_DESCRIPTOR *desc;
    CUdeviceptr dptr;
    size_t Pitch;
} cuTexRefSetAddress2D_v3_params;

typedef struct cuTexRefSetFormat_params_st {
    CUtexref hTexRef;
    CUarray_format fmt;
    int NumPackedComponents;
} cuTexRefSetFormat_params;

typedef struct cuTexRefSetAddressMode_params_st {
    CUtexref hTexRef;
    int dim;
    CUaddress_mode am;
} cuTexRefSetAddressMode_params;

typedef struct cuTexRefSetFilterMode_params_st {
    CUtexref hTexRef;
    CUfilter_mode fm;
} cuTexRefSetFilterMode_params;

typedef struct cuTexRefSetMipmapFilterMode_params_st {
    CUtexref hTexRef;
    CUfilter_mode fm;
} cuTexRefSetMipmapFilterMode_params;

typedef struct cuTexRefSetMipmapLevelBias_params_st {
    CUtexref hTexRef;
    float bias;
} cuTexRefSetMipmapLevelBias_params;

typedef struct cuTexRefSetMipmapLevelClamp_params_st {
    CUtexref hTexRef;
    float minMipmapLevelClamp;
    float maxMipmapLevelClamp;
} cuTexRefSetMipmapLevelClamp_params;

typedef struct cuTexRefSetMaxAnisotropy_params_st {
    CUtexref hTexRef;
    unsigned int maxAniso;
} cuTexRefSetMaxAnisotropy_params;

typedef struct cuTexRefSetBorderColor_params_st {
    CUtexref hTexRef;
    float *pBorderColor;
} cuTexRefSetBorderColor_params;

typedef struct cuTexRefSetFlags_params_st {
    CUtexref hTexRef;
    unsigned int Flags;
} cuTexRefSetFlags_params;

typedef struct cuTexRefGetAddress_v2_params_st {
    CUdeviceptr *pdptr;
    CUtexref hTexRef;
} cuTexRefGetAddress_v2_params;

typedef struct cuTexRefGetArray_params_st {
    CUarray *phArray;
    CUtexref hTexRef;
} cuTexRefGetArray_params;

typedef struct cuTexRefGetMipmappedArray_params_st {
    CUmipmappedArray *phMipmappedArray;
    CUtexref hTexRef;
} cuTexRefGetMipmappedArray_params;

typedef struct cuTexRefGetAddressMode_params_st {
    CUaddress_mode *pam;
    CUtexref hTexRef;
    int dim;
} cuTexRefGetAddressMode_params;

typedef struct cuTexRefGetFilterMode_params_st {
    CUfilter_mode *pfm;
    CUtexref hTexRef;
} cuTexRefGetFilterMode_params;

typedef struct cuTexRefGetFormat_params_st {
    CUarray_format *pFormat;
    int *pNumChannels;
    CUtexref hTexRef;
} cuTexRefGetFormat_params;

typedef struct cuTexRefGetMipmapFilterMode_params_st {
    CUfilter_mode *pfm;
    CUtexref hTexRef;
} cuTexRefGetMipmapFilterMode_params;

typedef struct cuTexRefGetMipmapLevelBias_params_st {
    float *pbias;
    CUtexref hTexRef;
} cuTexRefGetMipmapLevelBias_params;

typedef struct cuTexRefGetMipmapLevelClamp_params_st {
    float *pminMipmapLevelClamp;
    float *pmaxMipmapLevelClamp;
    CUtexref hTexRef;
} cuTexRefGetMipmapLevelClamp_params;

typedef struct cuTexRefGetMaxAnisotropy_params_st {
    int *pmaxAniso;
    CUtexref hTexRef;
} cuTexRefGetMaxAnisotropy_params;

typedef struct cuTexRefGetBorderColor_params_st {
    float *pBorderColor;
    CUtexref hTexRef;
} cuTexRefGetBorderColor_params;

typedef struct cuTexRefGetFlags_params_st {
    unsigned int *pFlags;
    CUtexref hTexRef;
} cuTexRefGetFlags_params;

typedef struct cuTexRefCreate_params_st {
    CUtexref *pTexRef;
} cuTexRefCreate_params;

typedef struct cuTexRefDestroy_params_st {
    CUtexref hTexRef;
} cuTexRefDestroy_params;

typedef struct cuSurfRefSetArray_params_st {
    CUsurfref hSurfRef;
    CUarray hArray;
    unsigned int Flags;
} cuSurfRefSetArray_params;

typedef struct cuSurfRefGetArray_params_st {
    CUarray *phArray;
    CUsurfref hSurfRef;
} cuSurfRefGetArray_params;

typedef struct cuTexObjectCreate_params_st {
    CUtexObject *pTexObject;
    const CUDA_RESOURCE_DESC *pResDesc;
    const CUDA_TEXTURE_DESC *pTexDesc;
    const CUDA_RESOURCE_VIEW_DESC *pResViewDesc;
} cuTexObjectCreate_params;

typedef struct cuTexObjectDestroy_params_st {
    CUtexObject texObject;
} cuTexObjectDestroy_params;

typedef struct cuTexObjectGetResourceDesc_params_st {
    CUDA_RESOURCE_DESC *pResDesc;
    CUtexObject texObject;
} cuTexObjectGetResourceDesc_params;

typedef struct cuTexObjectGetTextureDesc_params_st {
    CUDA_TEXTURE_DESC *pTexDesc;
    CUtexObject texObject;
} cuTexObjectGetTextureDesc_params;

typedef struct cuTexObjectGetResourceViewDesc_params_st {
    CUDA_RESOURCE_VIEW_DESC *pResViewDesc;
    CUtexObject texObject;
} cuTexObjectGetResourceViewDesc_params;

typedef struct cuSurfObjectCreate_params_st {
    CUsurfObject *pSurfObject;
    const CUDA_RESOURCE_DESC *pResDesc;
} cuSurfObjectCreate_params;

typedef struct cuSurfObjectDestroy_params_st {
    CUsurfObject surfObject;
} cuSurfObjectDestroy_params;

typedef struct cuSurfObjectGetResourceDesc_params_st {
    CUDA_RESOURCE_DESC *pResDesc;
    CUsurfObject surfObject;
} cuSurfObjectGetResourceDesc_params;

typedef struct cuDeviceCanAccessPeer_params_st {
    int *canAccessPeer;
    CUdevice dev;
    CUdevice peerDev;
} cuDeviceCanAccessPeer_params;

typedef struct cuCtxEnablePeerAccess_params_st {
    CUcontext peerContext;
    unsigned int Flags;
} cuCtxEnablePeerAccess_params;

typedef struct cuCtxDisablePeerAccess_params_st {
    CUcontext peerContext;
} cuCtxDisablePeerAccess_params;

typedef struct cuDeviceGetP2PAttribute_params_st {
    int *value;
    CUdevice_P2PAttribute attrib;
    CUdevice srcDevice;
    CUdevice dstDevice;
} cuDeviceGetP2PAttribute_params;

typedef struct cuGraphicsUnregisterResource_params_st {
    CUgraphicsResource resource;
} cuGraphicsUnregisterResource_params;

typedef struct cuGraphicsSubResourceGetMappedArray_params_st {
    CUarray *pArray;
    CUgraphicsResource resource;
    unsigned int arrayIndex;
    unsigned int mipLevel;
} cuGraphicsSubResourceGetMappedArray_params;

typedef struct cuGraphicsResourceGetMappedMipmappedArray_params_st {
    CUmipmappedArray *pMipmappedArray;
    CUgraphicsResource resource;
} cuGraphicsResourceGetMappedMipmappedArray_params;

typedef struct cuGraphicsResourceGetMappedPointer_v2_params_st {
    CUdeviceptr *pDevPtr;
    size_t *pSize;
    CUgraphicsResource resource;
} cuGraphicsResourceGetMappedPointer_v2_params;

typedef struct cuGraphicsResourceSetMapFlags_v2_params_st {
    CUgraphicsResource resource;
    unsigned int flags;
} cuGraphicsResourceSetMapFlags_v2_params;

typedef struct cuGraphicsMapResources_ptsz_params_st {
    unsigned int count;
    CUgraphicsResource *resources;
    CUstream hStream;
} cuGraphicsMapResources_ptsz_params;

typedef struct cuGraphicsUnmapResources_ptsz_params_st {
    unsigned int count;
    CUgraphicsResource *resources;
    CUstream hStream;
} cuGraphicsUnmapResources_ptsz_params;

typedef struct cuGetExportTable_params_st {
    const void **ppExportTable;
    const CUuuid *pExportTableId;
} cuGetExportTable_params;

typedef struct cuFuncGetModule_params_st {
    CUmodule *hmod;
    CUfunction hfunc;
} cuFuncGetModule_params;

typedef struct cuMemHostRegister_params_st {
    void *p;
    size_t bytesize;
    unsigned int Flags;
} cuMemHostRegister_params;

typedef struct cuGraphicsResourceSetMapFlags_params_st {
    CUgraphicsResource resource;
    unsigned int flags;
} cuGraphicsResourceSetMapFlags_params;

typedef struct cuLinkCreate_params_st {
    unsigned int numOptions;
    CUjit_option *options;
    void **optionValues;
    CUlinkState *stateOut;
} cuLinkCreate_params;

typedef struct cuLinkAddData_params_st {
    CUlinkState state;
    CUjitInputType type;
    void *data;
    size_t size;
    const char *name;
    unsigned int numOptions;
    CUjit_option *options;
    void **optionValues;
} cuLinkAddData_params;

typedef struct cuLinkAddFile_params_st {
    CUlinkState state;
    CUjitInputType type;
    const char *path;
    unsigned int numOptions;
    CUjit_option *options;
    void **optionValues;
} cuLinkAddFile_params;

typedef struct cuTexRefSetAddress2D_v2_params_st {
    CUtexref hTexRef;
    const CUDA_ARRAY_DESCRIPTOR *desc;
    CUdeviceptr dptr;
    size_t Pitch;
} cuTexRefSetAddress2D_v2_params;

typedef struct cuDeviceTotalMem_params_st {
    unsigned int *bytes;
    CUdevice dev;
} cuDeviceTotalMem_params;

typedef struct cuCtxCreate_params_st {
    CUcontext *pctx;
    unsigned int flags;
    CUdevice dev;
} cuCtxCreate_params;

typedef struct cuModuleGetGlobal_params_st {
    CUdeviceptr_v1 *dptr;
    unsigned int *bytes;
    CUmodule hmod;
    const char *name;
} cuModuleGetGlobal_params;

typedef struct cuMemGetInfo_params_st {
    unsigned int *free;
    unsigned int *total;
} cuMemGetInfo_params;

typedef struct cuMemAlloc_params_st {
    CUdeviceptr_v1 *dptr;
    unsigned int bytesize;
} cuMemAlloc_params;

typedef struct cuMemAllocPitch_params_st {
    CUdeviceptr_v1 *dptr;
    unsigned int *pPitch;
    unsigned int WidthInBytes;
    unsigned int Height;
    unsigned int ElementSizeBytes;
} cuMemAllocPitch_params;

typedef struct cuMemFree_params_st {
    CUdeviceptr_v1 dptr;
} cuMemFree_params;

typedef struct cuMemGetAddressRange_params_st {
    CUdeviceptr_v1 *pbase;
    unsigned int *psize;
    CUdeviceptr_v1 dptr;
} cuMemGetAddressRange_params;

typedef struct cuMemAllocHost_params_st {
    void **pp;
    unsigned int bytesize;
} cuMemAllocHost_params;

typedef struct cuMemHostGetDevicePointer_params_st {
    CUdeviceptr_v1 *pdptr;
    void *p;
    unsigned int Flags;
} cuMemHostGetDevicePointer_params;

typedef struct cuMemcpyHtoD_params_st {
    CUdeviceptr_v1 dstDevice;
    const void *srcHost;
    unsigned int ByteCount;
} cuMemcpyHtoD_params;

typedef struct cuMemcpyDtoH_params_st {
    void *dstHost;
    CUdeviceptr_v1 srcDevice;
    unsigned int ByteCount;
} cuMemcpyDtoH_params;

typedef struct cuMemcpyDtoD_params_st {
    CUdeviceptr_v1 dstDevice;
    CUdeviceptr_v1 srcDevice;
    unsigned int ByteCount;
} cuMemcpyDtoD_params;

typedef struct cuMemcpyDtoA_params_st {
    CUarray dstArray;
    unsigned int dstOffset;
    CUdeviceptr_v1 srcDevice;
    unsigned int ByteCount;
} cuMemcpyDtoA_params;

typedef struct cuMemcpyAtoD_params_st {
    CUdeviceptr_v1 dstDevice;
    CUarray srcArray;
    unsigned int srcOffset;
    unsigned int ByteCount;
} cuMemcpyAtoD_params;

typedef struct cuMemcpyHtoA_params_st {
    CUarray dstArray;
    unsigned int dstOffset;
    const void *srcHost;
    unsigned int ByteCount;
} cuMemcpyHtoA_params;

typedef struct cuMemcpyAtoH_params_st {
    void *dstHost;
    CUarray srcArray;
    unsigned int srcOffset;
    unsigned int ByteCount;
} cuMemcpyAtoH_params;

typedef struct cuMemcpyAtoA_params_st {
    CUarray dstArray;
    unsigned int dstOffset;
    CUarray srcArray;
    unsigned int srcOffset;
    unsigned int ByteCount;
} cuMemcpyAtoA_params;

typedef struct cuMemcpyHtoAAsync_params_st {
    CUarray dstArray;
    unsigned int dstOffset;
    const void *srcHost;
    unsigned int ByteCount;
    CUstream hStream;
} cuMemcpyHtoAAsync_params;

typedef struct cuMemcpyAtoHAsync_params_st {
    void *dstHost;
    CUarray srcArray;
    unsigned int srcOffset;
    unsigned int ByteCount;
    CUstream hStream;
} cuMemcpyAtoHAsync_params;

typedef struct cuMemcpy2D_params_st {
    const CUDA_MEMCPY2D_v1 *pCopy;
} cuMemcpy2D_params;

typedef struct cuMemcpy2DUnaligned_params_st {
    const CUDA_MEMCPY2D_v1 *pCopy;
} cuMemcpy2DUnaligned_params;

typedef struct cuMemcpy3D_params_st {
    const CUDA_MEMCPY3D_v1 *pCopy;
} cuMemcpy3D_params;

typedef struct cuMemcpyHtoDAsync_params_st {
    CUdeviceptr_v1 dstDevice;
    const void *srcHost;
    unsigned int ByteCount;
    CUstream hStream;
} cuMemcpyHtoDAsync_params;

typedef struct cuMemcpyDtoHAsync_params_st {
    void *dstHost;
    CUdeviceptr_v1 srcDevice;
    unsigned int ByteCount;
    CUstream hStream;
} cuMemcpyDtoHAsync_params;

typedef struct cuMemcpyDtoDAsync_params_st {
    CUdeviceptr_v1 dstDevice;
    CUdeviceptr_v1 srcDevice;
    unsigned int ByteCount;
    CUstream hStream;
} cuMemcpyDtoDAsync_params;

typedef struct cuMemcpy2DAsync_params_st {
    const CUDA_MEMCPY2D_v1 *pCopy;
    CUstream hStream;
} cuMemcpy2DAsync_params;

typedef struct cuMemcpy3DAsync_params_st {
    const CUDA_MEMCPY3D_v1 *pCopy;
    CUstream hStream;
} cuMemcpy3DAsync_params;

typedef struct cuMemsetD8_params_st {
    CUdeviceptr_v1 dstDevice;
    unsigned char uc;
    unsigned int N;
} cuMemsetD8_params;

typedef struct cuMemsetD16_params_st {
    CUdeviceptr_v1 dstDevice;
    unsigned short us;
    unsigned int N;
} cuMemsetD16_params;

typedef struct cuMemsetD32_params_st {
    CUdeviceptr_v1 dstDevice;
    unsigned int ui;
    unsigned int N;
} cuMemsetD32_params;

typedef struct cuMemsetD2D8_params_st {
    CUdeviceptr_v1 dstDevice;
    unsigned int dstPitch;
    unsigned char uc;
    unsigned int Width;
    unsigned int Height;
} cuMemsetD2D8_params;

typedef struct cuMemsetD2D16_params_st {
    CUdeviceptr_v1 dstDevice;
    unsigned int dstPitch;
    unsigned short us;
    unsigned int Width;
    unsigned int Height;
} cuMemsetD2D16_params;

typedef struct cuMemsetD2D32_params_st {
    CUdeviceptr_v1 dstDevice;
    unsigned int dstPitch;
    unsigned int ui;
    unsigned int Width;
    unsigned int Height;
} cuMemsetD2D32_params;

typedef struct cuArrayCreate_params_st {
    CUarray *pHandle;
    const CUDA_ARRAY_DESCRIPTOR_v1 *pAllocateArray;
} cuArrayCreate_params;

typedef struct cuArrayGetDescriptor_params_st {
    CUDA_ARRAY_DESCRIPTOR_v1 *pArrayDescriptor;
    CUarray hArray;
} cuArrayGetDescriptor_params;

typedef struct cuArray3DCreate_params_st {
    CUarray *pHandle;
    const CUDA_ARRAY3D_DESCRIPTOR_v1 *pAllocateArray;
} cuArray3DCreate_params;

typedef struct cuArray3DGetDescriptor_params_st {
    CUDA_ARRAY3D_DESCRIPTOR_v1 *pArrayDescriptor;
    CUarray hArray;
} cuArray3DGetDescriptor_params;

typedef struct cuTexRefSetAddress_params_st {
    unsigned int *ByteOffset;
    CUtexref hTexRef;
    CUdeviceptr_v1 dptr;
    unsigned int bytes;
} cuTexRefSetAddress_params;

typedef struct cuTexRefSetAddress2D_params_st {
    CUtexref hTexRef;
    const CUDA_ARRAY_DESCRIPTOR_v1 *desc;
    CUdeviceptr_v1 dptr;
    unsigned int Pitch;
} cuTexRefSetAddress2D_params;

typedef struct cuTexRefGetAddress_params_st {
    CUdeviceptr_v1 *pdptr;
    CUtexref hTexRef;
} cuTexRefGetAddress_params;

typedef struct cuGraphicsResourceGetMappedPointer_params_st {
    CUdeviceptr_v1 *pDevPtr;
    unsigned int *pSize;
    CUgraphicsResource resource;
} cuGraphicsResourceGetMappedPointer_params;

typedef struct cuCtxDestroy_params_st {
    CUcontext ctx;
} cuCtxDestroy_params;

typedef struct cuCtxPopCurrent_params_st {
    CUcontext *pctx;
} cuCtxPopCurrent_params;

typedef struct cuCtxPushCurrent_params_st {
    CUcontext ctx;
} cuCtxPushCurrent_params;

typedef struct cuStreamDestroy_params_st {
    CUstream hStream;
} cuStreamDestroy_params;

typedef struct cuEventDestroy_params_st {
    CUevent hEvent;
} cuEventDestroy_params;

typedef struct cuDevicePrimaryCtxRelease_params_st {
    CUdevice dev;
} cuDevicePrimaryCtxRelease_params;

typedef struct cuDevicePrimaryCtxReset_params_st {
    CUdevice dev;
} cuDevicePrimaryCtxReset_params;

typedef struct cuDevicePrimaryCtxSetFlags_params_st {
    CUdevice dev;
    unsigned int flags;
} cuDevicePrimaryCtxSetFlags_params;

typedef struct cuMemcpyHtoD_v2_params_st {
    CUdeviceptr dstDevice;
    const void *srcHost;
    size_t ByteCount;
} cuMemcpyHtoD_v2_params;

typedef struct cuMemcpyDtoH_v2_params_st {
    void *dstHost;
    CUdeviceptr srcDevice;
    size_t ByteCount;
} cuMemcpyDtoH_v2_params;

typedef struct cuMemcpyDtoD_v2_params_st {
    CUdeviceptr dstDevice;
    CUdeviceptr srcDevice;
    size_t ByteCount;
} cuMemcpyDtoD_v2_params;

typedef struct cuMemcpyDtoA_v2_params_st {
    CUarray dstArray;
    size_t dstOffset;
    CUdeviceptr srcDevice;
    size_t ByteCount;
} cuMemcpyDtoA_v2_params;

typedef struct cuMemcpyAtoD_v2_params_st {
    CUdeviceptr dstDevice;
    CUarray srcArray;
    size_t srcOffset;
    size_t ByteCount;
} cuMemcpyAtoD_v2_params;

typedef struct cuMemcpyHtoA_v2_params_st {
    CUarray dstArray;
    size_t dstOffset;
    const void *srcHost;
    size_t ByteCount;
} cuMemcpyHtoA_v2_params;

typedef struct cuMemcpyAtoH_v2_params_st {
    void *dstHost;
    CUarray srcArray;
    size_t srcOffset;
    size_t ByteCount;
} cuMemcpyAtoH_v2_params;

typedef struct cuMemcpyAtoA_v2_params_st {
    CUarray dstArray;
    size_t dstOffset;
    CUarray srcArray;
    size_t srcOffset;
    size_t ByteCount;
} cuMemcpyAtoA_v2_params;

typedef struct cuMemcpyHtoAAsync_v2_params_st {
    CUarray dstArray;
    size_t dstOffset;
    const void *srcHost;
    size_t ByteCount;
    CUstream hStream;
} cuMemcpyHtoAAsync_v2_params;

typedef struct cuMemcpyAtoHAsync_v2_params_st {
    void *dstHost;
    CUarray srcArray;
    size_t srcOffset;
    size_t ByteCount;
    CUstream hStream;
} cuMemcpyAtoHAsync_v2_params;

typedef struct cuMemcpy2D_v2_params_st {
    const CUDA_MEMCPY2D *pCopy;
} cuMemcpy2D_v2_params;

typedef struct cuMemcpy2DUnaligned_v2_params_st {
    const CUDA_MEMCPY2D *pCopy;
} cuMemcpy2DUnaligned_v2_params;

typedef struct cuMemcpy3D_v2_params_st {
    const CUDA_MEMCPY3D *pCopy;
} cuMemcpy3D_v2_params;

typedef struct cuMemcpyHtoDAsync_v2_params_st {
    CUdeviceptr dstDevice;
    const void *srcHost;
    size_t ByteCount;
    CUstream hStream;
} cuMemcpyHtoDAsync_v2_params;

typedef struct cuMemcpyDtoHAsync_v2_params_st {
    void *dstHost;
    CUdeviceptr srcDevice;
    size_t ByteCount;
    CUstream hStream;
} cuMemcpyDtoHAsync_v2_params;

typedef struct cuMemcpyDtoDAsync_v2_params_st {
    CUdeviceptr dstDevice;
    CUdeviceptr srcDevice;
    size_t ByteCount;
    CUstream hStream;
} cuMemcpyDtoDAsync_v2_params;

typedef struct cuMemcpy2DAsync_v2_params_st {
    const CUDA_MEMCPY2D *pCopy;
    CUstream hStream;
} cuMemcpy2DAsync_v2_params;

typedef struct cuMemcpy3DAsync_v2_params_st {
    const CUDA_MEMCPY3D *pCopy;
    CUstream hStream;
} cuMemcpy3DAsync_v2_params;

typedef struct cuMemsetD8_v2_params_st {
    CUdeviceptr dstDevice;
    unsigned char uc;
    size_t N;
} cuMemsetD8_v2_params;

typedef struct cuMemsetD16_v2_params_st {
    CUdeviceptr dstDevice;
    unsigned short us;
    size_t N;
} cuMemsetD16_v2_params;

typedef struct cuMemsetD32_v2_params_st {
    CUdeviceptr dstDevice;
    unsigned int ui;
    size_t N;
} cuMemsetD32_v2_params;

typedef struct cuMemsetD2D8_v2_params_st {
    CUdeviceptr dstDevice;
    size_t dstPitch;
    unsigned char uc;
    size_t Width;
    size_t Height;
} cuMemsetD2D8_v2_params;

typedef struct cuMemsetD2D16_v2_params_st {
    CUdeviceptr dstDevice;
    size_t dstPitch;
    unsigned short us;
    size_t Width;
    size_t Height;
} cuMemsetD2D16_v2_params;

typedef struct cuMemsetD2D32_v2_params_st {
    CUdeviceptr dstDevice;
    size_t dstPitch;
    unsigned int ui;
    size_t Width;
    size_t Height;
} cuMemsetD2D32_v2_params;

typedef struct cuMemcpy_params_st {
    CUdeviceptr dst;
    CUdeviceptr src;
    size_t ByteCount;
} cuMemcpy_params;

typedef struct cuMemcpyAsync_params_st {
    CUdeviceptr dst;
    CUdeviceptr src;
    size_t ByteCount;
    CUstream hStream;
} cuMemcpyAsync_params;

typedef struct cuMemcpyPeer_params_st {
    CUdeviceptr dstDevice;
    CUcontext dstContext;
    CUdeviceptr srcDevice;
    CUcontext srcContext;
    size_t ByteCount;
} cuMemcpyPeer_params;

typedef struct cuMemcpyPeerAsync_params_st {
    CUdeviceptr dstDevice;
    CUcontext dstContext;
    CUdeviceptr srcDevice;
    CUcontext srcContext;
    size_t ByteCount;
    CUstream hStream;
} cuMemcpyPeerAsync_params;

typedef struct cuMemcpy3DPeer_params_st {
    const CUDA_MEMCPY3D_PEER *pCopy;
} cuMemcpy3DPeer_params;

typedef struct cuMemcpy3DPeerAsync_params_st {
    const CUDA_MEMCPY3D_PEER *pCopy;
    CUstream hStream;
} cuMemcpy3DPeerAsync_params;

typedef struct cuMemsetD8Async_params_st {
    CUdeviceptr dstDevice;
    unsigned char uc;
    size_t N;
    CUstream hStream;
} cuMemsetD8Async_params;

typedef struct cuMemsetD16Async_params_st {
    CUdeviceptr dstDevice;
    unsigned short us;
    size_t N;
    CUstream hStream;
} cuMemsetD16Async_params;

typedef struct cuMemsetD32Async_params_st {
    CUdeviceptr dstDevice;
    unsigned int ui;
    size_t N;
    CUstream hStream;
} cuMemsetD32Async_params;

typedef struct cuMemsetD2D8Async_params_st {
    CUdeviceptr dstDevice;
    size_t dstPitch;
    unsigned char uc;
    size_t Width;
    size_t Height;
    CUstream hStream;
} cuMemsetD2D8Async_params;

typedef struct cuMemsetD2D16Async_params_st {
    CUdeviceptr dstDevice;
    size_t dstPitch;
    unsigned short us;
    size_t Width;
    size_t Height;
    CUstream hStream;
} cuMemsetD2D16Async_params;

typedef struct cuMemsetD2D32Async_params_st {
    CUdeviceptr dstDevice;
    size_t dstPitch;
    unsigned int ui;
    size_t Width;
    size_t Height;
    CUstream hStream;
} cuMemsetD2D32Async_params;

typedef struct cuStreamGetPriority_params_st {
    CUstream hStream;
    int *priority;
} cuStreamGetPriority_params;

typedef struct cuStreamGetFlags_params_st {
    CUstream hStream;
    unsigned int *flags;
} cuStreamGetFlags_params;

typedef struct cuStreamGetCtx_params_st {
    CUstream hStream;
    CUcontext *pctx;
} cuStreamGetCtx_params;

typedef struct cuStreamWaitEvent_params_st {
    CUstream hStream;
    CUevent hEvent;
    unsigned int Flags;
} cuStreamWaitEvent_params;

typedef struct cuStreamAddCallback_params_st {
    CUstream hStream;
    CUstreamCallback callback;
    void *userData;
    unsigned int flags;
} cuStreamAddCallback_params;

typedef struct cuStreamAttachMemAsync_params_st {
    CUstream hStream;
    CUdeviceptr dptr;
    size_t length;
    unsigned int flags;
} cuStreamAttachMemAsync_params;

typedef struct cuStreamQuery_params_st {
    CUstream hStream;
} cuStreamQuery_params;

typedef struct cuStreamSynchronize_params_st {
    CUstream hStream;
} cuStreamSynchronize_params;

typedef struct cuEventRecord_params_st {
    CUevent hEvent;
    CUstream hStream;
} cuEventRecord_params;

typedef struct cuLaunchKernel_params_st {
    CUfunction f;
    unsigned int gridDimX;
    unsigned int gridDimY;
    unsigned int gridDimZ;
    unsigned int blockDimX;
    unsigned int blockDimY;
    unsigned int blockDimZ;
    unsigned int sharedMemBytes;
    CUstream hStream;
    void **kernelParams;
    void **extra;
} cuLaunchKernel_params;

typedef struct cuLaunchHostFunc_params_st {
    CUstream hStream;
    CUhostFn fn;
    void *userData;
} cuLaunchHostFunc_params;

typedef struct cuGraphicsMapResources_params_st {
    unsigned int count;
    CUgraphicsResource *resources;
    CUstream hStream;
} cuGraphicsMapResources_params;

typedef struct cuGraphicsUnmapResources_params_st {
    unsigned int count;
    CUgraphicsResource *resources;
    CUstream hStream;
} cuGraphicsUnmapResources_params;

typedef struct cuStreamWriteValue32_params_st {
    CUstream stream;
    CUdeviceptr addr;
    cuuint32_t value;
    unsigned int flags;
} cuStreamWriteValue32_params;

typedef struct cuStreamWaitValue32_params_st {
    CUstream stream;
    CUdeviceptr addr;
    cuuint32_t value;
    unsigned int flags;
} cuStreamWaitValue32_params;

typedef struct cuStreamWriteValue64_params_st {
    CUstream stream;
    CUdeviceptr addr;
    cuuint64_t value;
    unsigned int flags;
} cuStreamWriteValue64_params;

typedef struct cuStreamWaitValue64_params_st {
    CUstream stream;
    CUdeviceptr addr;
    cuuint64_t value;
    unsigned int flags;
} cuStreamWaitValue64_params;

typedef struct cuStreamBatchMemOp_params_st {
    CUstream stream;
    unsigned int count;
    CUstreamBatchMemOpParams *paramArray;
    unsigned int flags;
} cuStreamBatchMemOp_params;

typedef struct cuMemPrefetchAsync_params_st {
    CUdeviceptr devPtr;
    size_t count;
    CUdevice dstDevice;
    CUstream hStream;
} cuMemPrefetchAsync_params;

typedef struct cuLaunchCooperativeKernel_params_st {
    CUfunction f;
    unsigned int gridDimX;
    unsigned int gridDimY;
    unsigned int gridDimZ;
    unsigned int blockDimX;
    unsigned int blockDimY;
    unsigned int blockDimZ;
    unsigned int sharedMemBytes;
    CUstream hStream;
    void **kernelParams;
} cuLaunchCooperativeKernel_params;

typedef struct cuSignalExternalSemaphoresAsync_params_st {
    const CUexternalSemaphore *extSemArray;
    const CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS *paramsArray;
    unsigned int numExtSems;
    CUstream stream;
} cuSignalExternalSemaphoresAsync_params;

typedef struct cuWaitExternalSemaphoresAsync_params_st {
    const CUexternalSemaphore *extSemArray;
    const CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS *paramsArray;
    unsigned int numExtSems;
    CUstream stream;
} cuWaitExternalSemaphoresAsync_params;

typedef struct cuStreamBeginCapture_params_st {
    CUstream hStream;
} cuStreamBeginCapture_params;

typedef struct cuStreamBeginCapture_ptsz_params_st {
    CUstream hStream;
} cuStreamBeginCapture_ptsz_params;

typedef struct cuStreamBeginCapture_v2_params_st {
    CUstream hStream;
    CUstreamCaptureMode mode;
} cuStreamBeginCapture_v2_params;

typedef struct cuStreamEndCapture_params_st {
    CUstream hStream;
    CUgraph *phGraph;
} cuStreamEndCapture_params;

typedef struct cuStreamIsCapturing_params_st {
    CUstream hStream;
    CUstreamCaptureStatus *captureStatus;
} cuStreamIsCapturing_params;

typedef struct cuStreamGetCaptureInfo_params_st {
    CUstream hStream;
    CUstreamCaptureStatus *captureStatus;
    cuuint64_t *id;
} cuStreamGetCaptureInfo_params;

typedef struct cuGraphLaunch_params_st {
    CUgraphExec hGraph;
    CUstream hStream;
} cuGraphLaunch_params;

typedef struct cuStreamCopyAttributes_params_st {
    CUstream dstStream;
    CUstream srcStream;
} cuStreamCopyAttributes_params;

typedef struct cuStreamGetAttribute_params_st {
    CUstream hStream;
    CUstreamAttrID attr;
    CUstreamAttrValue *value;
} cuStreamGetAttribute_params;

typedef struct cuStreamSetAttribute_params_st {
    CUstream hStream;
    CUstreamAttrID attr;
    const CUstreamAttrValue *param;
} cuStreamSetAttribute_params;

typedef struct cuIpcOpenMemHandle_params_st {
    CUdeviceptr *pdptr;
    CUipcMemHandle handle;
    unsigned int Flags;
} cuIpcOpenMemHandle_params;

typedef struct cuGraphInstantiate_params_st {
    CUgraphExec *phGraphExec;
    CUgraph hGraph;
    CUgraphNode *phErrorNode;
    char *logBuffer;
    size_t bufferSize;
} cuGraphInstantiate_params;

#endif  // file guard
