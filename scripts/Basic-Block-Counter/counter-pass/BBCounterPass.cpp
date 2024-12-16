#include "llvm/Pass.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/LegacyPassManager.h"

#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/IPO/PassManagerBuilder.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"

#include <algorithm>
#include <vector>
#include <string>
#include <iostream>

using namespace llvm;
using namespace std;

namespace {
  struct BBCounterPass : public FunctionPass  {
    static char ID;
    unsigned int bbNum = 0;

    BBCounterPass() : FunctionPass (ID) {}
    virtual bool runOnFunction(Function &Func);
    virtual bool doFinalization(Module &M);
  };
}

bool BBCounterPass::runOnFunction(Function &Func) {
  // Get the function to call from our runtime library.
  bool modified = false;
  LLVMContext &Ctx = Func.getContext();
  ArrayRef<Type *> paramTypes = {Type::getInt32Ty(Ctx)}; // Parameter type
  Type *retType = Type::getVoidTy(Ctx); // Return type
  FunctionType *instFuncType = FunctionType::get(retType, paramTypes, false); // Instrumentation function type
  Module *Mod = Func.getParent();
  FunctionCallee countFunc = Mod->getOrInsertFunction("bbCounter", instFuncType);
  errs() << "\nFunction: " << Func.getName() << "\n";
  for (auto &BB : Func) {
    string bb_name = Func.getName().str();
    bb_name = std::to_string(bbNum) + "-" + bb_name + "-" + BB.getName().str();
    errs() << "BB: " << bbNum << ", Name: " << bb_name << "\t\tN. of Inst: " << BB.size() << "\n";

    IRBuilder<> builder(&BB);
    builder.SetInsertPoint(BB.getFirstNonPHI()); // Insert before first non-PHI
    Value *args[] = {builder.getInt32(bbNum)};
    // Value *args = builder.CreateGlobalStringPtr(bb_name.c_str());
    // Value *args[] = {builder.getInt32(100), strVal};
    builder.CreateCall(countFunc, args);
    modified |= true;
    bbNum++;
  }
  return modified;
}

bool BBCounterPass::doFinalization(Module &Mod) {
  bool modified = false;
  errs() << "\nBBCounterPass::doFinalization:- Total number of basic blocks in source files:  " << bbNum << "\n";
  for (auto &Func : Mod) {
    if (Func.getName() == "main") {
      LLVMContext &Ctx = Func.getContext();
      ArrayRef<Type *> paramTypes = {Type::getInt32Ty(Ctx)}; // Parameter type
      Type *retType = Type::getVoidTy(Ctx); // Return type
      FunctionType *instFuncType = FunctionType::get(retType, paramTypes, false); // Instrumentation function type
      IRBuilder<> builder(Ctx);

      BasicBlock *entryBB = &Func.getEntryBlock();
      errs() << "\n\ninitCounter inserted in the beginning of BB: " << entryBB->getName() << "\n";
      FunctionCallee initFunc = Mod.getOrInsertFunction("initCounter", instFuncType);
      builder.SetInsertPoint(entryBB->getFirstNonPHI());
      Value *initargs[] = {builder.getInt32(bbNum)};
      builder.CreateCall(initFunc, initargs);

      BasicBlock *exitBB = &Func.getBasicBlockList().back();
      errs() << "dumpCounts inserted in the end of BB: " << entryBB->getName() << "\n";
      FunctionCallee dumpFunc = Mod.getOrInsertFunction("dumpCounts", instFuncType);
      builder.SetInsertPoint(exitBB->getTerminator());
      Value *termiargs[] = {builder.getInt32(0)};
      builder.CreateCall(dumpFunc, termiargs);

      modified |= true;
    }
  }
  return modified;
}

char BBCounterPass::ID = 0;

static void registerBBCounterPass(const PassManagerBuilder &, legacy::PassManagerBase &PM)
{
  PM.add(new BBCounterPass());
}
static RegisterStandardPasses
    RegisterMyPass(PassManagerBuilder::EP_EarlyAsPossible,
                   registerBBCounterPass);
