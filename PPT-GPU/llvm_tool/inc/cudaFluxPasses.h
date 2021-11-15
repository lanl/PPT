#ifndef PROJECT_BLOCKPROFILER_H
#define PROJECT_BLOCKPROFILER_H

#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Module.h"
#include "llvm/Pass.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#include "llvm/Support/YAMLParser.h"
#include "llvm/Support/YAMLTraits.h"

/// DeviceBlockProfilerPass
/// Instruments nvptx kernels with code to count the execution count of each
/// basic block
struct FluxDevicePass : public llvm::ModulePass {
public:
  static char ID;
  FluxDevicePass() : llvm::ModulePass(ID) {}

  virtual bool runOnModule(llvm::Module &M) override;

  virtual void getAnalysisUsage(llvm::AnalysisUsage &AU) const override;

  void releaseMemory() override;
};

/// HostBlockProfilerPass
/// Provides setup for DeviceBlockProfilerPass in host code
struct FluxHostPass : public llvm::ModulePass {
public:
  static char ID;
  FluxHostPass() : llvm::ModulePass(ID) {}

  virtual bool runOnModule(llvm::Module &M) override;

  virtual void getAnalysisUsage(llvm::AnalysisUsage &AU) const override;

  void releaseMemory() override;
};

#endif // PROJECT_BLOCKPROFILER_H
