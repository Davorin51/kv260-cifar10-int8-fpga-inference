// ==============================================================
// Vitis HLS - High-Level Synthesis from C, C++ and OpenCL v2025.2 (64-bit)
// Tool Version Limit: 2025.11
// Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
// Copyright 2022-2025 Advanced Micro Devices, Inc. All Rights Reserved.
// 
// ==============================================================
#ifndef XCIFAR10_CNN_H
#define XCIFAR10_CNN_H

#ifdef __cplusplus
extern "C" {
#endif

/***************************** Include Files *********************************/
#ifndef __linux__
#include "xil_types.h"
#include "xil_assert.h"
#include "xstatus.h"
#include "xil_io.h"
#else
#include <stdint.h>
#include <assert.h>
#include <dirent.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>
#include <stddef.h>
#endif
#include "xcifar10_cnn_hw.h"

/**************************** Type Definitions ******************************/
#ifdef __linux__
typedef uint8_t u8;
typedef uint16_t u16;
typedef uint32_t u32;
typedef uint64_t u64;
#else
typedef struct {
#ifdef SDT
    char *Name;
#else
    u16 DeviceId;
#endif
    u64 Control_BaseAddress;
} XCifar10_cnn_Config;
#endif

typedef struct {
    u64 Control_BaseAddress;
    u32 IsReady;
} XCifar10_cnn;

typedef u32 word_type;

/***************** Macros (Inline Functions) Definitions *********************/
#ifndef __linux__
#define XCifar10_cnn_WriteReg(BaseAddress, RegOffset, Data) \
    Xil_Out32((BaseAddress) + (RegOffset), (u32)(Data))
#define XCifar10_cnn_ReadReg(BaseAddress, RegOffset) \
    Xil_In32((BaseAddress) + (RegOffset))
#else
#define XCifar10_cnn_WriteReg(BaseAddress, RegOffset, Data) \
    *(volatile u32*)((BaseAddress) + (RegOffset)) = (u32)(Data)
#define XCifar10_cnn_ReadReg(BaseAddress, RegOffset) \
    *(volatile u32*)((BaseAddress) + (RegOffset))

#define Xil_AssertVoid(expr)    assert(expr)
#define Xil_AssertNonvoid(expr) assert(expr)

#define XST_SUCCESS             0
#define XST_DEVICE_NOT_FOUND    2
#define XST_OPEN_DEVICE_FAILED  3
#define XIL_COMPONENT_IS_READY  1
#endif

/************************** Function Prototypes *****************************/
#ifndef __linux__
#ifdef SDT
int XCifar10_cnn_Initialize(XCifar10_cnn *InstancePtr, UINTPTR BaseAddress);
XCifar10_cnn_Config* XCifar10_cnn_LookupConfig(UINTPTR BaseAddress);
#else
int XCifar10_cnn_Initialize(XCifar10_cnn *InstancePtr, u16 DeviceId);
XCifar10_cnn_Config* XCifar10_cnn_LookupConfig(u16 DeviceId);
#endif
int XCifar10_cnn_CfgInitialize(XCifar10_cnn *InstancePtr, XCifar10_cnn_Config *ConfigPtr);
#else
int XCifar10_cnn_Initialize(XCifar10_cnn *InstancePtr, const char* InstanceName);
int XCifar10_cnn_Release(XCifar10_cnn *InstancePtr);
#endif

void XCifar10_cnn_Start(XCifar10_cnn *InstancePtr);
u32 XCifar10_cnn_IsDone(XCifar10_cnn *InstancePtr);
u32 XCifar10_cnn_IsIdle(XCifar10_cnn *InstancePtr);
u32 XCifar10_cnn_IsReady(XCifar10_cnn *InstancePtr);
void XCifar10_cnn_EnableAutoRestart(XCifar10_cnn *InstancePtr);
void XCifar10_cnn_DisableAutoRestart(XCifar10_cnn *InstancePtr);

void XCifar10_cnn_Set_image_r(XCifar10_cnn *InstancePtr, u64 Data);
u64 XCifar10_cnn_Get_image_r(XCifar10_cnn *InstancePtr);
void XCifar10_cnn_Set_logits(XCifar10_cnn *InstancePtr, u64 Data);
u64 XCifar10_cnn_Get_logits(XCifar10_cnn *InstancePtr);

void XCifar10_cnn_InterruptGlobalEnable(XCifar10_cnn *InstancePtr);
void XCifar10_cnn_InterruptGlobalDisable(XCifar10_cnn *InstancePtr);
void XCifar10_cnn_InterruptEnable(XCifar10_cnn *InstancePtr, u32 Mask);
void XCifar10_cnn_InterruptDisable(XCifar10_cnn *InstancePtr, u32 Mask);
void XCifar10_cnn_InterruptClear(XCifar10_cnn *InstancePtr, u32 Mask);
u32 XCifar10_cnn_InterruptGetEnabled(XCifar10_cnn *InstancePtr);
u32 XCifar10_cnn_InterruptGetStatus(XCifar10_cnn *InstancePtr);

#ifdef __cplusplus
}
#endif

#endif
