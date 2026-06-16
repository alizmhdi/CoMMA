/*
 * Copyright (c) 2025 Google LLC All rights reserved.
 * Use of this source code is governed by a BSD-style
 * license that can be found in the LICENSE file.
 */

#ifndef NCCL_NET_H_
#define NCCL_NET_H_

#include <stdint.h>

#ifdef __cplusplus
extern "C"
{
#endif

typedef struct ncclNetProperties_v7_t ncclNetProperties_t;
typedef struct ncclNet_v7_t ncclNet_t;
typedef struct ncclCollNet ncclCollNet_t;

typedef enum
{
    ncclSuccess = 0,
    ncclUnhandledCudaError = 1,
    ncclSystemError = 2,
    ncclInternalError = 3,
    ncclInvalidArgument = 4,
    ncclInvalidUsage = 5,
    ncclRemoteError = 6,
    ncclInProgress = 7,
    ncclNumResults = 8
} ncclResult_t;

typedef enum
{
    NCCL_LOG_NONE = 0,
    NCCL_LOG_VERSION = 1,
    NCCL_LOG_WARN = 2,
    NCCL_LOG_INFO = 3,
    NCCL_LOG_ABORT = 4,
    NCCL_LOG_TRACE = 5
} ncclDebugLogLevel;

typedef void (*ncclDebugLogger_t)(ncclDebugLogLevel level, unsigned long flags, const char *file, int line, const char *fmt, ...);

typedef enum
{
    NCCL_INIT = 1,
    NCCL_COLL = 2,
    NCCL_P2P = 4,
    NCCL_SHM = 8,
    NCCL_NET = 16,
    NCCL_GRAPH = 32,
    NCCL_TUNING = 64,
    NCCL_ENV = 128,
    NCCL_ALLOC = 256,
    NCCL_CALL = 512,
    NCCL_PROXY = 1024,
    NCCL_NVLS = 2048,
    NCCL_ALL = ~0
} ncclDebugLogSubSys;

#ifdef __cplusplus
}
#endif

#endif  // NCCL_NET_H_
