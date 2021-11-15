#ifndef IRUTILS_H
#define IRUTILS_H

#include "BasicBlockEnumerator.h"
#include <PTXFunction.h>

#include <llvm/IR/Instructions.h>
#include <llvm/IR/Module.h>
#include <llvm/IRReader/IRReader.h>
#include <llvm/Linker/Linker.h>
#include <llvm/Support/Casting.h>
// instead of:
//#include <llvm/IR/TypeBuilder.h>
// adapt with llvm version?
#include <llvm/IR/IRBuilder.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/SourceMgr.h>

#include <string>

namespace mekong {

// Utility Functions

std::string getModulePrefix(llvm::Module *m);
void dumpModuleToFile(llvm::Module &m, std::string filepath);
void dumpModuleToFile(llvm::Module &m, const char *filepath);
std::map<llvm::BasicBlock *, int>
getBlockIDMap(llvm::Function *func, std::vector<mekong::PTXFunction> funcVec,
              std::string originalFunctionName);

// Analysis Functions

// Transformation Functions

llvm::GlobalVariable *createGlobalString(llvm::Module &M, llvm::StringRef Str,
                                         const llvm::Twine &Name = "",
                                         unsigned AddressSpace = 0);

llvm::Constant *createGlobalStringPtr(llvm::Module &M, llvm::StringRef Str,
                                      const llvm::Twine &Name = "",
                                      unsigned AddressSpace = 0);

void linkIR(llvm::StringRef ir, llvm::Module &m);

llvm::Function *getPrintfFunc(llvm::Module &module);

llvm::CallInst *callPrintf(llvm::Module &module, std::string str,
                           llvm::Value *val);
llvm::CallInst *callPrintf(llvm::Module &module, llvm::IRBuilder<> &builder,
                           std::string str, llvm::Value *val);

} // namespace mekong

#endif
