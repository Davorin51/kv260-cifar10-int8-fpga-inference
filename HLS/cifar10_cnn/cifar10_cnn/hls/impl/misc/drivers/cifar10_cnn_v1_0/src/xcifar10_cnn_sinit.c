// ==============================================================
// Vitis HLS - High-Level Synthesis from C, C++ and OpenCL v2025.2 (64-bit)
// Tool Version Limit: 2025.11
// Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
// Copyright 2022-2025 Advanced Micro Devices, Inc. All Rights Reserved.
// 
// ==============================================================
#ifndef __linux__

#include "xstatus.h"
#ifdef SDT
#include "xparameters.h"
#endif
#include "xcifar10_cnn.h"

extern XCifar10_cnn_Config XCifar10_cnn_ConfigTable[];

#ifdef SDT
XCifar10_cnn_Config *XCifar10_cnn_LookupConfig(UINTPTR BaseAddress) {
	XCifar10_cnn_Config *ConfigPtr = NULL;

	int Index;

	for (Index = (u32)0x0; XCifar10_cnn_ConfigTable[Index].Name != NULL; Index++) {
		if (!BaseAddress || XCifar10_cnn_ConfigTable[Index].Control_BaseAddress == BaseAddress) {
			ConfigPtr = &XCifar10_cnn_ConfigTable[Index];
			break;
		}
	}

	return ConfigPtr;
}

int XCifar10_cnn_Initialize(XCifar10_cnn *InstancePtr, UINTPTR BaseAddress) {
	XCifar10_cnn_Config *ConfigPtr;

	Xil_AssertNonvoid(InstancePtr != NULL);

	ConfigPtr = XCifar10_cnn_LookupConfig(BaseAddress);
	if (ConfigPtr == NULL) {
		InstancePtr->IsReady = 0;
		return (XST_DEVICE_NOT_FOUND);
	}

	return XCifar10_cnn_CfgInitialize(InstancePtr, ConfigPtr);
}
#else
XCifar10_cnn_Config *XCifar10_cnn_LookupConfig(u16 DeviceId) {
	XCifar10_cnn_Config *ConfigPtr = NULL;

	int Index;

	for (Index = 0; Index < XPAR_XCIFAR10_CNN_NUM_INSTANCES; Index++) {
		if (XCifar10_cnn_ConfigTable[Index].DeviceId == DeviceId) {
			ConfigPtr = &XCifar10_cnn_ConfigTable[Index];
			break;
		}
	}

	return ConfigPtr;
}

int XCifar10_cnn_Initialize(XCifar10_cnn *InstancePtr, u16 DeviceId) {
	XCifar10_cnn_Config *ConfigPtr;

	Xil_AssertNonvoid(InstancePtr != NULL);

	ConfigPtr = XCifar10_cnn_LookupConfig(DeviceId);
	if (ConfigPtr == NULL) {
		InstancePtr->IsReady = 0;
		return (XST_DEVICE_NOT_FOUND);
	}

	return XCifar10_cnn_CfgInitialize(InstancePtr, ConfigPtr);
}
#endif

#endif

