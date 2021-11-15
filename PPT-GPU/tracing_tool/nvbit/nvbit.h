/* Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#pragma once
#include <stdint.h>
#include <stdio.h>
#include <cassert>
#include <iomanip>  // std::setw
#include <iostream>
#include <string>
#include <vector>

#include "instr_types.h"

#include "tools_cuda_api_meta.h"
#define __CUDA_API_VERSION_INTERNAL
#include "cuda.h"
#include "generated_cuda_meta.h"

#define NVBIT_VERSION "1.5.1"

/* Instruction class returned by the NVBit inspection API nvbit_get_instrs */
class Instr {
  public:
    /* returns the "string"  containing the SASS, i.e. IMAD.WIDE R8, R8, R9 */
    const char* getSass();
    /* returns offset in bytes of this instruction within the function */
    uint32_t getOffset();
    /* returns the id of the instruction within the function */
    uint32_t getIdx();
    /* returns true if instruction used predicate */
    bool hasPred();
    /* returns predicate number, only valid if hasPred() == true */
    int getPredNum();
    /* returns true if predicate is negated (i.e. @!P0), only valid if hasPred()
     * == true */
    bool isPredNeg();
    /* if predicate is uniform predicate (e.g., @UP0), only valid if hasPred()
     * == true */
    bool isPredUniform();
    /* returns full opcode of the instruction (i.e. IMAD.WIDE ) */
    const char* getOpcode();
    /* returns short opcode of the instruction (i.e. IMAD.WIDE returns IMAD) */
    const char* getOpcodeShort();

    /* returns MemorySpace type (was getMemOpType) */
    InstrType::MemorySpace getMemorySpace();
    bool isLoad();
    bool isStore();
    bool isExtended();
    int getSize();


    /* get number of operands */
    int getNumOperands();
    /* get specific operand */
    const InstrType::operand_t* getOperand(int num_operand);

    /* print fully decoded instruction */
    void printDecoded();
    /* prints one line instruction with idx, offset, sass */
    void print(const char* prefix = NULL);

  private:
    /* Constructor used internally by NVBit */
    Instr(const char* sass);
    /* Reserved variable used internally by NVBit */
    const void* reserved;
    friend class Nvbit;
    friend class Function;
};

/* basic block struct */
typedef struct { std::vector<Instr*> instrs; } basic_block_t;

/* control flow graph struct */
typedef struct {
    /* indicates the control flow graph can't be statically predetermined
     * because the function from which is belong uses jmx/brx types of branches
     * which targets depends of registers values that are known only
     * at runtime */
    bool is_degenerate;
    /* vector of basic block */
    std::vector<basic_block_t*> bbs;
} CFG_t;

/*
 * callback definitions as enumerated in tools_cuda_api_meta.h
 */
#define DEFINE_ENUM_CBID_API_CUDA(area, id, name, params) API_CUDA_##name,
typedef enum {
    API_CUDA_INVALID = 0,
    CU_TOOLS_FOR_EACH_CUDA_API_FUNC(DEFINE_ENUM_CBID_API_CUDA)
} nvbit_api_cuda_t;

extern "C" {

/*********************************************************************
 *
 *                  NVBit tool callbacks
 *     (implement these functions to get a callback from NVBit)
 *
 **********************************************************************/
/* This function is called as soon as the program starts, no GPU calls
 * should be made a this moment */
void nvbit_at_init();

/* This function is called just before the program terminates, no GPU calls
 * should be made a this moment */
void nvbit_at_term();

/* This function is called as soon as a GPU context is started and it should
 * contain any code that we would like to execute at that moment. */
void nvbit_at_ctx_init(CUcontext ctx);

/* This function is called as soon as the GPU context is terminated and it
 * should contain any code that we would like to execute at that moment. */
void nvbit_at_ctx_term(CUcontext ctx);

/* This is the function called every beginning (is_exit = 0) and
 * end (is_exit = 1) of a CUDA driver call.
 * cbid identifies the CUDA driver call as specified by the enum
 * nvbit_api_cuda_t, see tools_cuda_api_meta.h for the list of cbid.
 * Name is its the driver call name.
 * params is pointer to* one of the structures defined in the
 * generated_cuda_meta.h.
 * Params must be casted to the correct struct based on the cbid.
 * For instance if cbid = cuMemcpyDtoH_v2 then params must be casted to
 * (cuMemcpyDtoH_v2_params *)
 * */
void nvbit_at_cuda_event(CUcontext ctx, int is_exit, nvbit_api_cuda_t cbid,
                         const char* event_name, void* params,
                         CUresult* pStatus);

/*********************************************************************
 *
 *          NVBit inspection APIs  (provided by NVBit)
 *
 **********************************************************************/
/* Get vector of related functions */
std::vector<CUfunction> nvbit_get_related_functions(CUcontext ctx,
                                                    CUfunction func);

/* Get vector of instructions composing the CUfunction */
const std::vector<Instr*>& nvbit_get_instrs(CUcontext ctx, CUfunction func);

/* Get control flow graph (CFG) */
const CFG_t& nvbit_get_CFG(CUcontext ctx, CUfunction func);

/* Allows to get a function name from its CUfunction */
const char* nvbit_get_func_name(CUcontext ctx, CUfunction f,
                                bool mangled = false);

/* Get line information for a particular instruction offset if available,
 * binary must be compiled with --generate-line-info   (-lineinfo) */
bool nvbit_get_line_info(CUcontext cuctx, CUfunction cufunc, uint32_t offset,
                         char** file_name, char** dir_name, uint32_t* line);

/* Get the SM family */
uint32_t nvbit_get_sm_family(CUcontext cuctx);

/* Allows to get PC address of the function */
uint64_t nvbit_get_func_addr(CUfunction func);

/* Returns true if function is a kernel (i.e. __global__ ) */
bool nvbit_is_func_kernel(CUcontext ctx, CUfunction func);

/* Allows to get shmem base address from CUcontext
 * shmem range is [shmem_base_addr, shmem_base_addr+16MB) and
 * the base address is 16MB aligned.  */
uint64_t nvbit_get_shmem_base_addr(CUcontext cuctx);

/* Allows to get local memory base address from CUcontext
 * local mem range is [shmem_base_addr, shmem_base_addr+16MB) and
 * the base address is 16MB aligned.  */
uint64_t nvbit_get_local_mem_base_addr(CUcontext cuctx);

/*********************************************************************
 *
 *          NVBit injection APIs  (provided by NVBit)
 *
 **********************************************************************/

/* Enumeration used by nvbit_insert_call to specify where we want to insert
 * the device function for a given Instr, if before or after */
typedef enum { IPOINT_BEFORE, IPOINT_AFTER } ipoint_t;

/* This function inserts a device function call named "dev_func_name",
 * before or after Instr (ipoint_t { IPOINT_BEFORE, IPOINT_AFTER}).
 * It is important to remember that calls to device functions are
 * identified by name (as opposed to function pointers) and that
 * we declare the function as:
 *
 *        extern "C" __device__ __noinline__
 *
 * to prevent the compiler from optimizing out this device function
 * during compilation.
 *
 * Multiple device functions can be inserted before or after and the
 * order in which they get executed is defined by the order in which
 * they have been inserted. */

void nvbit_insert_call(const Instr* instr, const char* dev_func_name,
                       ipoint_t point);

/* Add int32_t argument to last injected call, value of the (uniform) predicate
 * for this instruction */
void nvbit_add_call_arg_guard_pred_val(const Instr* instr,
                                       bool is_variadic_arg = false);

/* Add int32_t argument to last injected call, value of the designated predicate
 * for this instruction */
void nvbit_add_call_arg_pred_val_at(const Instr* instr, int pred_num,
                                    bool is_variadic_arg = false);

/* Add int32_t argument to last injected call, value of the designated uniform
 * predicate for this instruction */
void nvbit_add_call_arg_upred_val_at(const Instr* instr, int upred_num,
                                     bool is_variadic_arg = false);

/* Add int32_t argument to last injected call, value of the entire predicate
 * register for this thread */
void nvbit_add_call_arg_pred_reg(const Instr* instr,
                                 bool is_variadic_arg = false);

/* Add int32_t argument to last injected call, value of the entire uniform
 * predicate register for this thread */
void nvbit_add_call_arg_upred_reg(const Instr* instr,
                                  bool is_variadic_arg = false);

/* Add uint32_t argument to last injected call, constant 32-bit value */
void nvbit_add_call_arg_const_val32(const Instr* instr, uint32_t val,
                                    bool is_variadic_arg = false);

/* Add uint64_t argument to last injected call, constant 64-bit value */
void nvbit_add_call_arg_const_val64(const Instr* instr, uint64_t val,
                                    bool is_variadic_arg = false);

/* Add uint32_t argument to last injected call, content of the register reg_num
 */
void nvbit_add_call_arg_reg_val(const Instr* instr, int reg_num,
                                bool is_variadic_arg = false);

/* Add uint32_t argument to last injected call, content of the
 * uniform register reg_num */
void nvbit_add_call_arg_ureg_val(const Instr* instr, int reg_num,
                                 bool is_variadic_arg = false);

/* Add uint32_t argument to last injected call, 32-bit at launch value at offset
 * "offset", set at launch time with nvbit_set_at_launch */
void nvbit_add_call_arg_launch_val32(const Instr* instr, int offset,
                                     bool is_variadic_arg = false);

/* Add uint64_t argument to last injected call, 64-bit at launch value at offset
 * "offset", set at launch time with nvbit_set_at_launch */
void nvbit_add_call_arg_launch_val64(const Instr* instr, int offset,
                                     bool is_variadic_arg = false);

/* Add uint32_t argument to last injected call, constant bank value at
 * c[bankid][bankoffset] */
void nvbit_add_call_arg_cbank_val(const Instr* instr, int bankid,
                                  int bankoffset, bool is_variadic_arg = false);

/* The 64-bit memory reference address accessed by this instruction
  Typically memory instructions have only 1 MREF so in general id = 0 */
void nvbit_add_call_arg_mref_addr64(const Instr* instr, int id = 0,
                                    bool is_variadic_arg = false);

/* Remove the original instruction */
void nvbit_remove_orig(const Instr* instr);

/*********************************************************************
 *
 *          NVBit device level APIs  (provided by NVBit)
 *
 **********************************************************************/

#ifdef __CUDACC__
/* device function used to read/write register values
 * writes are permanent into application state */
__device__ __noinline__ int32_t nvbit_read_reg(uint64_t reg_num);
__device__ __noinline__ void nvbit_write_reg(uint64_t reg_num, int32_t reg_val);
__device__ __noinline__ int32_t nvbit_read_ureg(uint64_t reg_num);
__device__ __noinline__ void nvbit_write_ureg(uint64_t reg_num,
                                              int32_t reg_val);
__device__ __noinline__ int32_t nvbit_read_pred_reg(void);
__device__ __noinline__ void nvbit_write_pred_reg(int32_t reg_val);
__device__ __noinline__ int32_t nvbit_read_upred_reg(void);
__device__ __noinline__ void nvbit_write_upred_reg(int32_t reg_val);
#endif

/*********************************************************************
 *
 *          NVBit control APIs  (provided by NVBit)
 *
 **********************************************************************/

/* Run instrumented on original function (and its related functions)
 * based on flag value */
void nvbit_enable_instrumented(CUcontext ctx, CUfunction func, bool flag,
                               bool apply_to_related = true);

/* Set arguments at launch time, that will be loaded on input argument of
 * the instrumentation function */
void nvbit_set_at_launch(CUcontext ctx, CUfunction func, void* buf,
                         uint32_t nbytes);

/* Notify nvbit of a pthread used by the tool, this pthread will not
 * trigger any call backs even if executing CUDA events of kernel launches.
 * Multiple pthreads can be registered one after the other. */
void nvbit_set_tool_pthread(pthread_t tool_pthread);
void nvbit_unset_tool_pthread(pthread_t tool_pthread);

/* Set nvdisasm */
void nvbit_set_nvdisasm(const char* nvdisasm);
}

/*********************************************************************
 *
 *              Macros to read environment variables
 *
 **********************************************************************/

#define PRINT_VAR(env_var, help, var)                                      \
    std::cout << std::setw(20) << env_var << " = " << var << " - " << help \
              << std::endl;

#define GET_VAR_INT(var, env_var, def, help) \
    if (getenv(env_var)) {                   \
        var = atoi(getenv(env_var));         \
    } else {                                 \
        var = def;                           \
    }                                        \
    PRINT_VAR(env_var, help, var)

#define GET_VAR_LONG(var, env_var, def, help) \
    if (getenv(env_var)) {                    \
        var = atol(getenv(env_var));          \
    } else {                                  \
        var = def;                            \
    }                                         \
    PRINT_VAR(env_var, help, var)

#define GET_VAR_STR(var, env_var, help) \
    if (getenv(env_var)) {              \
        std::string s(getenv(env_var)); \
        var = s;                        \
    }                                   \
    PRINT_VAR(env_var, help, var)
