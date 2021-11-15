#ifndef HOSTUTILS_H
#define HOSTUTILS_H

#include <llvm/IR/Instructions.h>
#include <llvm/IR/Module.h>
#include <vector>

namespace mekong {

// Structs, Classes, Datatypes
struct KernelDescriptor {
  llvm::Function *handle;
  llvm::StringRef name;
};

// Analysis Functions
bool usesNewKernelLaunch(llvm::Module &m);
void getKernelDescriptors(llvm::Module &m, std::vector<KernelDescriptor> &desc);
void getKernelLaunchSites(llvm::Function *klFun, std::vector<llvm::CallBase *> &callSites);
void getKernelArguments(llvm::CallBase *kernelLaunchSite, std::vector<llvm::Value *> &args);
llvm::CallBase *getKernelConfigCall(llvm::Module &m, llvm::CallBase *kernelLaunchSite);
void getKernelLaunchConfig(llvm::Module &m, llvm::CallBase *kernelLaunchSite, std::vector<llvm::Value *> &config);

llvm::Function *getCudaSynchronizeStream(llvm::Module &m);

// Transformation Functions
llvm::Value *createCudaGlobalVar(llvm::Module &m, const std::string name, llvm::Type *varType);
void registerKernel(llvm::Module &m, const std::string name, llvm::Function *kernelWrapper);
llvm::CallBase *replaceKernelLaunch(llvm::Module &m, llvm::CallBase *kernelLaunchSite,
                                    llvm::Function *replacementWrapper,
                                    std::vector<llvm::Value *> &additionalArguments);
llvm::Function *createKernelWrapper(llvm::Module &m, const std::string name, llvm::FunctionType *ft);

} // namespace mekong

#endif
