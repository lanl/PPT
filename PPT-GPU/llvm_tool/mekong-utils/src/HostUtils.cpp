#include "Mekong-Utils.h"

#include <llvm/Support/Casting.h>
// instead of:
//#include <llvm/IR/TypeBuilder.h>
// adapt with llvm version?
#include "llvm/IR/CFG.h"
#include "llvm/Support/VersionTuple.h"
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/Module.h>
#include <llvm/Support/raw_ostream.h>

#include <queue>

using namespace llvm;
using namespace std;

bool launchFinder(BasicBlock *block, void *launchCallPtr) {
  CallBase **ptr = (CallBase **)launchCallPtr;

  for (Instruction &inst : *block) {

    // If inst is a call inst look for cuda funktions
    if (CallBase *ci = dyn_cast_or_null<CallBase>(&inst)) {
      StringRef name = ci->getCalledFunction()->getName();
      if (name == "cudaSetupArgument")
        continue;
      if (name == "cudaLaunch") {
        *ptr = ci;
        return true;
        // Any other call could be the kernel wrapper
        // Check if cudaLaunch is called from the possible kernel wrapper
        // function
      } else {

        Function *cudaLaunch = block->getParent()->getParent()->getFunction("cudaLaunch");
        for (Value *launchCallVal : cudaLaunch->users()) {
          if (CallBase *launchCallBase = dyn_cast_or_null<CallBase>(launchCallVal)) {
            if (launchCallBase->getFunction() == ci->getCalledFunction()) {
              *ptr = launchCallBase;
              return true;
            }
          }
        } // end for (Value* launchCallVal : cudaLaunch->users())

      } // end else
    }   // end if (CallBase *ci = dync_cast_or_null<CallBase>(&inst))
    if (InvokeInst *invi = dyn_cast_or_null<InvokeInst>(&inst)) {
      // Check if cudaLaunch is called from the possible kernel wrapper function
      Function *cudaLaunch = block->getParent()->getParent()->getFunction("cudaLaunch");
      for (Value *launchCallVal : cudaLaunch->users()) {
        if (CallBase *launchCallBase = dyn_cast_or_null<CallBase>(launchCallVal)) {
          if (launchCallBase->getFunction() == invi->getCalledFunction()) {
            *ptr = launchCallBase;
            return true;
          }
        }
      } // end for (Value* launchCallVal : cudaLaunch->users())

    } // end if (CallBase *ci = dync_cast_or_null<CallBase>(&inst))

  } // end for (Insturction &inst : *block)

  return false;
}

namespace mekong {

///===------------------------------------------------------------------===//
///                           Analysis Functions
///===------------------------------------------------------------------===//

bool usesNewKernelLaunch(llvm::Module &m) { return m.getSDKVersion() >= VersionTuple(9, 2); }

void getKernelDescriptors(llvm::Module &m, std::vector<KernelDescriptor> &desc) {
  Function *registerFunction = m.getFunction("__cudaRegisterFunction");
  if (registerFunction == nullptr)
    return;
  for (auto *user : registerFunction->users()) {
    // check for callbase as this will find all call sites of "__cudaRegisterFunction"
    CallBase *callBase = dyn_cast_or_null<CallBase>(user);
    if (callBase != nullptr) {
      // 2nd argumen is the handle under which the kernel is registered
      Value *val_func = (callBase->getArgOperand(1)->stripPointerCastsAndAliases());
      Function *handle = dyn_cast_or_null<Function>(val_func);
      // 3rd argument is the kernel name (on device side)
      GlobalVariable *gv = dyn_cast_or_null<GlobalVariable>(callBase->getArgOperand(2)->stripPointerCastsAndAliases());
      ConstantDataSequential *val_name = nullptr;
      if (gv != nullptr)
        val_name = dyn_cast_or_null<ConstantDataSequential>(gv->getInitializer());

      if (val_func != nullptr && val_name != nullptr) {
        StringRef name = val_name->getAsString();
        // Remove null byte at end
        name = name.substr(0, name.size() - 1);
        desc.push_back({handle, name});
      }
    }
  }
  return;
}

// Finds the instructions that call the kernel launch wrapper function (klFun)
// in this function cudaLaunch or cudaLaunchKernel will be called
// this function will not return the call sites if the wrapper function is
// already inlined
void getKernelLaunchSites(llvm::Function *klFun, std::vector<llvm::CallBase *> &callSites) {
  for (auto *user : klFun->users()) {
    CallBase *callBase = dyn_cast_or_null<CallBase>(user);
    // user must be callbase and call the kernel launch function
    if (callBase != nullptr && (callBase->getCalledFunction() == klFun)) {
      callSites.push_back(callBase);
    }
  }
  return;
}

void getKernelArguments(llvm::CallBase *kernelLaunchSite, std::vector<llvm::Value *> &args) {
  for (Use &op : kernelLaunchSite->operands()) {
    // Make sure not to inclide the kernel launch wrapper
    Function *fun = dyn_cast_or_null<Function>(op.get());
    if (fun != nullptr and fun == kernelLaunchSite->getCalledFunction())
      continue;
    // also skip invoke operands which are basicblocks
    if (dyn_cast_or_null<BasicBlock>(op.get()) != nullptr)
      continue;

    args.push_back(op.get());
  }
  // get arguments regarding launch configuration
  return;
}

CallBase *getKernelConfigCall(llvm::Module &m, llvm::CallBase *kernelLaunchSite) {
  Function *confFunc = nullptr;

  if (usesNewKernelLaunch(m)) {
    confFunc = m.getFunction("__cudaPushCallConfiguration");
  } else {
    confFunc = m.getFunction("cudaConfigureCall");
  }

  assert(confFunc != nullptr);

  // BFSearch back to configuration call
  CallBase *confCall = nullptr;
  queue<BasicBlock *> q;
  q.push(kernelLaunchSite->getParent());
  while (q.empty() == false and confCall == nullptr) {
    BasicBlock *currentBlock = q.front();
    q.pop();

    if (currentBlock != kernelLaunchSite->getParent()) {
      // Search for configure call
      for (Instruction &inst : *currentBlock) {
        CallBase *ci = dyn_cast_or_null<CallBase>(&inst);
        if (ci != nullptr && ci->getCalledFunction() == confFunc) {
          confCall = ci;
          break;
        }
      }
    }

    // Add predecessors if call was not found
    pred_iterator PI = pred_begin(currentBlock), E = pred_end(currentBlock);
    assert(PI != E or confCall != nullptr && "Could not find kernel configuration call!");
    for (; PI != E; ++PI) {
      q.push(*PI);
    }
  }

  assert(confCall != nullptr && "Kernel configuration call not found!");

  return confCall;
}

void getKernelLaunchConfig(llvm::Module &m, llvm::CallBase *kernelLaunchSite, std::vector<llvm::Value *> &config) {
  CallBase *confCall = getKernelConfigCall(m, kernelLaunchSite);

  for (Use &val : confCall->operands()) {
    // only the arguments are wanted not the function itself
    if (dyn_cast_or_null<Function>(val.get()) != nullptr)
      continue;
    // discard basic blocks in case the callbase is an invoke instruction
    if (dyn_cast_or_null<BasicBlock>(val.get()) != nullptr)
      continue;
    config.push_back(val.get());
  }
  return;
}

llvm::Function *getCudaSynchronizeStream(llvm::Module &m) {
  LLVMContext &ctx = m.getContext();
  FunctionType *cudaSyncStream_ft =
      FunctionType::get(Type::getInt32Ty(ctx), {m.getTypeByName("struct.CUstream_st")->getPointerTo()}, false);
  return dyn_cast<Function>(m.getOrInsertFunction("cudaStreamSynchronize", cudaSyncStream_ft).getCallee());
}

void getGridConfig(llvm::CallBase *call, llvm::Value *(&arguments)[4]) {
  assert(call->getCalledFunction()->getName() == "cudaLaunchKernel");
  arguments[0] = call->getArgOperand(1);
  arguments[1] = call->getArgOperand(2);
  arguments[2] = call->getArgOperand(3);
  arguments[3] = call->getArgOperand(3);
}

///===------------------------------------------------------------------===//
///                        Transformation Functions
///===------------------------------------------------------------------===//

void registerKernel(llvm::Module &m, const std::string name, llvm::Function *kernelWrapper) {
  LLVMContext &ctx = m.getContext();

  Function *registerGlobals = m.getFunction("__cuda_register_globals");

  assert(registerGlobals != nullptr && "Could not find __cuda_register_globals!");
  Function *registerFunction = m.getFunction("__cudaRegisterFunction");
  assert(registerFunction != nullptr && "Could not find __cudaRegisterFunction!");

  // find cuda binary handle
  Value *bin_handle = nullptr;

  if (registerGlobals != nullptr) {
    bin_handle = &*registerGlobals->arg_begin();

  } else { // due to the asser this should never be reached
    registerGlobals = m.getFunction("__cuda_module_ctor");

    for (auto &bb : *registerGlobals) {
      for (auto &inst : bb) {
        if (CallBase *ci = dyn_cast_or_null<CallBase>(&inst)) {
          if (ci->getCalledFunction()->getName() == "__cudaRegisterFatBinary") {
            //                                       __cudaRegisterFatBinary
            bin_handle = ci;
          }
        }
      }
    }
  }

  assert(bin_handle != nullptr && "No cuda binary handle found!");

  IRBuilder<> builder(ctx);
  builder.SetInsertPoint(&registerGlobals->back().back());

  Value *wrapper_casted = builder.CreateBitCast(kernelWrapper, builder.getInt8PtrTy());
  Value *globalKernelNameString = builder.CreateGlobalStringPtr(name);
  Value *null = ConstantPointerNull::get(builder.getInt8PtrTy());
  Value *int32null = ConstantPointerNull::get(builder.getInt32Ty()->getPointerTo());
  vector<Value *> registerFunctionsArgs = {bin_handle,
                                           wrapper_casted,
                                           globalKernelNameString,
                                           globalKernelNameString,
                                           builder.getInt32(-1),
                                           null,
                                           null,
                                           null,
                                           null,
                                           int32null};

  // CallBase *errorCode =
  builder.CreateCall(registerFunction, registerFunctionsArgs);
  // mekong::callPrintf(m, "RegisterFunction: %d\n",
  // errorCode)->insertAfter(errorCode);
}

/// Creates a wrapper for a cuda kernel in module m with the given name
/// The functiontype is the the function signature of the kernel
Function *createKernelWrapper(llvm::Module &m, const std::string name, FunctionType *ft) {
  LLVMContext &ctx = m.getContext();

  IRBuilder<> builder(ctx);
  FunctionType *launchType =
      FunctionType::get(builder.getInt32Ty(),
                        {builder.getInt8PtrTy(), builder.getInt64Ty(), builder.getInt32Ty(), builder.getInt64Ty(),
                         builder.getInt32Ty(), builder.getInt8PtrTy()->getPointerTo(), builder.getInt64Ty(),
                         m.getTypeByName("struct.CUstream_st")->getPointerTo()},
                        false);
  // alternativ:
  // m.getFunction("cudaConfigureCall")->Type()-><LetzterTypderArgumente>

  Function *launchKernel = dyn_cast<Function>(m.getOrInsertFunction("cudaLaunchKernel", launchType).getCallee());

  // Same convention like the cuda api function: grid, and block sizes are i32
  // variables and the first two are merged to one i64
  vector<Type *> wrapperParams = {builder.getInt64Ty(), builder.getInt32Ty(),
                                  builder.getInt64Ty(), builder.getInt32Ty(),
                                  builder.getInt64Ty(), m.getTypeByName("struct.CUstream_st")->getPointerTo()};

  // add kernel parameters
  for (auto *param : ft->params())
    wrapperParams.push_back(param);

  FunctionType *wrapperType = FunctionType::get(builder.getInt32Ty(), wrapperParams, false);

  Function *wrapper = dyn_cast<Function>(m.getOrInsertFunction(name, wrapperType).getCallee());

  BasicBlock *entry = BasicBlock::Create(ctx, "entry", wrapper, nullptr);

  builder.SetInsertPoint(entry);

  // Get the 8 Arguments for cudaLaunchKernel (dim3 is translated to i64 + i32)
  // cudaLaunchKernel ( const void* func, dim3 gridDim, dim3 blockDim,
  //                    void** args, size_t sharedMem, cudaStream_t stream )
  vector<Value *> launchKernelArgs;

  // 1. function handle
  Value *fuptr = builder.CreatePointerCast(wrapper, builder.getInt8PtrTy());
  launchKernelArgs.push_back(fuptr);

  // 2.-5. gridDim and blockDim
  auto it = wrapper->arg_begin();
  // launchKernelArgs.push_back(gridXY);  // Skip Original Grid XY  Value with
  // fixed one
  launchKernelArgs.push_back(&*it);
  ++it;
  // Grid Z
  launchKernelArgs.push_back(&*it);
  ++it;
  // launchKernelArgs.push_back(blockXY);   // Skip original block XY value with
  // fixed one
  launchKernelArgs.push_back(&*it);
  ++it;
  launchKernelArgs.push_back(&*it);
  ++it;

  Value *shm = &*(it++);    // Shared Memory
  Value *stream = &*(it++); // Stream

  // 6. pointer to argument ptr array
  Value *ptr_array = builder.CreateAlloca(builder.getInt8PtrTy(), builder.getInt64(ft->params().size()));

  for (int i = 0; i < ft->params().size(); ++i) {
    // Get the parameter for the kernel, which start after the grid etc
    Value *argument = wrapper->arg_begin() + 6 + i;
    Value *memptr = builder.CreateAlloca(argument->getType());
    Value *store = builder.CreateStore(argument, memptr);
    Value *strptr = builder.CreateGEP(ptr_array, builder.getInt64(i));
    strptr = builder.CreatePointerCast(strptr, argument->getType()->getPointerTo()->getPointerTo());
    builder.CreateStore(memptr, strptr);
  }

  launchKernelArgs.push_back(ptr_array);

  // 7. + 8.
  launchKernelArgs.push_back(shm);
  launchKernelArgs.push_back(stream);

  Value *ret = builder.CreateCall(launchKernel, launchKernelArgs);

  builder.CreateRet(ret);

  return wrapper;
}

Function *createDummyKernelWrapper(llvm::Module &m, const std::string name) {
  LLVMContext &ctx = m.getContext();

  IRBuilder<> builder(ctx);
  FunctionType *fty = FunctionType::get(builder.getVoidTy(), {}, false);

  Function *dummy = dyn_cast<Function>(m.getOrInsertFunction(name, fty).getCallee());

  BasicBlock *entry = BasicBlock::Create(ctx, "entry", dummy, nullptr);

  builder.SetInsertPoint(entry);
  builder.CreateRet(builder.getInt32(0));

  return dummy;
}

llvm::CallBase *replaceKernelLaunch(llvm::Module &m, llvm::CallBase *kernelLaunchSite,
                                    llvm::Function *replacementWrapper,
                                    std::vector<llvm::Value *> &additionalArguments) {

  // gather launch config and kernel arguments
  std::vector<Value *> config;
  getKernelLaunchConfig(m, kernelLaunchSite, config);
  std::vector<Value *> kargs;
  getKernelArguments(kernelLaunchSite, kargs);
  CallBase *confCall = getKernelConfigCall(m, kernelLaunchSite);
  assert(confCall != nullptr);

  LLVMContext &ctx = m.getContext();
  IRBuilder<> builder(ctx);

  // check if config call is invoke inst and insert branch if so
  InvokeInst *confInv = dyn_cast_or_null<InvokeInst>(confCall);
  if (confInv != nullptr) {
    BasicBlock *branchTarget = confInv->getNormalDest();
    builder.SetInsertPoint(confCall);
    builder.CreateBr(branchTarget);
  }

  // replace all uses of confCall with 0 (for success) and erase it
  confCall->replaceAllUsesWith(builder.getInt32(0));
  confCall->eraseFromParent();

  // Prepare Launch Call Args
  std::vector<Value *> args;
  // append kernel launch config
  for (auto &val : config) {
    args.push_back(val);
  }
  // append kernel arguments
  for (auto &val : kargs) {
    args.push_back(val);
  }
  // append additional arguments
  for (auto &val : additionalArguments) {
    args.push_back(val);
  }

  // insert before old kernel launch
  builder.SetInsertPoint(kernelLaunchSite);

  FunctionType *ft = replacementWrapper->getFunctionType();
  auto params = ft->params();
  // cast pointer to stream if types do not match
  if (args[5]->getType() != params[5]) {
    args[5] = builder.CreateBitCast(args[5], params[5]);
  }

  // insert new kernel launch
  CallBase *newLaunchCall = builder.CreateCall(replacementWrapper, args);

  // check if kernelLaunch is invoke inst and insert branch if so
  InvokeInst *kernelInv = dyn_cast_or_null<InvokeInst>(kernelLaunchSite);
  if (kernelInv != nullptr) {
    BasicBlock *branchTarget = kernelInv->getNormalDest();
    builder.SetInsertPoint(kernelLaunchSite);
    builder.CreateBr(branchTarget);
  }

  // remove old kernel launch
  kernelLaunchSite->eraseFromParent();

  return newLaunchCall;
}

} // namespace mekong
