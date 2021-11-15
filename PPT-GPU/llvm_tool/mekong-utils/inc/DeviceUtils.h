#ifndef DEVICEUTILS_H
#define DEVICEUTILS_H

// STD C++ includes
#include <vector>

// LLVM includes
#include <llvm/IR/Module.h>

namespace mekong {

// Analysis Functions

void getKernels(llvm::Module &m, std::vector<llvm::Function *> &kernels);

// Transformation Functions

void registerGlobalVar(llvm::Module &m, std::string name, llvm::Type *type,
                       llvm::GlobalVariable *&gv);
void loadGlobalVar(llvm::Function *kernel, llvm::GlobalVariable *gv,
                   llvm::Value *&val);

llvm::Function *cloneAndAddArgs(llvm::Function *source,
                                std::vector<llvm::Type *> argType,
                                std::vector<std::string> name);

void markKernel(llvm::Module &m, llvm::Function *kernel);

} // namespace mekong

#endif
