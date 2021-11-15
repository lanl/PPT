#include "DeviceUtils.h"

#include <llvm/IR/Instructions.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/Transforms/Utils/Cloning.h>

using namespace llvm;
using namespace std;

namespace mekong {

///===------------------------------------------------------------------===//
///                           Analysis Functions
///===------------------------------------------------------------------===//

void getKernels(llvm::Module &m, std::vector<llvm::Function *> &kernels) {
  NamedMDNode *kernelMD = m.getNamedMetadata("nvvm.annotations");

  if (kernelMD) {
    for (const MDNode *node : kernelMD->operands()) {
      for (const MDOperand &op : node->operands()) {
        Metadata *md = op.get();
        if (ValueAsMetadata *v = dyn_cast_or_null<ValueAsMetadata>(md)) {
          if (Function *kernel = dyn_cast<Function>(v->getValue())) {
            assert(kernel != nullptr && "Kernel is nullptr");
            kernels.push_back(kernel);
          }
        }
      }
    }
  } else {
    errs() << "No kernel meta data found!\n";
  }
}

///===------------------------------------------------------------------===//
///                        Transformation Functions
///===------------------------------------------------------------------===//

llvm::Function *cloneAndAddArgs(llvm::Function *source,
                                std::vector<llvm::Type *> argType,
                                std::vector<std::string> name) {
  assert(source != nullptr && "Source Function to clone is null!");
  assert(not source->isDeclaration() && "Source Function is declaration!");

  vector<Type *> cloneArgs = source->getFunctionType()->params().vec();

  for (auto *arg : argType)
    cloneArgs.push_back(arg);

  FunctionType *cloneType =
      FunctionType::get(source->getReturnType(), cloneArgs, false);

  Function *clone =
      Function::Create(cloneType, source->getLinkage(),
                       source->getName() + "_clone", source->getParent());

  // link clone and source arguments
  ValueToValueMapTy vMap;

  auto ait = clone->arg_begin();
  for (Argument &arg : source->args()) {
    vMap.insert(std::make_pair(&arg, WeakTrackingVH(&(*ait))));
    // give the arguments the same name //
    ait->setName(arg.getName());
    ++ait;
  }

  auto nit = name.begin();
  while (ait != clone->arg_end()) {
    ait->setName(*nit);
    ++ait;
    ++nit;
  }

  SmallVector<ReturnInst *, 10> dummy;
  CloneFunctionInto(clone, source, vMap, true, dummy);

  return clone;
}

void registerGlobalVar(Module &m, string name, Type *type,
                       GlobalVariable *&gv) {
  LLVMContext &ctx = m.getContext();
  gv = new GlobalVariable(m, type, false, GlobalValue::ExternalLinkage, nullptr,
                          name, nullptr, GlobalVariable::NotThreadLocal, 1,
                          true);
  // gv->setAlignment(4);
  gv->setAlignment(MaybeAlign(4));
}

void loadGlobalVar(Function *kernel, GlobalVariable *gv, Value *&val) {
  Instruction *insertPoint = &*(kernel->begin()->begin());

  AddrSpaceCastInst *gvp =
      new AddrSpaceCastInst(gv, gv->getType()->getPointerTo(), "", insertPoint);
  Type *ty = cast<PointerType>(gvp->getType())->getElementType();
  val = new LoadInst(ty, gvp, "", insertPoint);
}

/// Adds metadata to the current module that will make the given function
/// callable as kernel
/// @param m[in] Module to add the metadata to
/// @param kernel[in] Function to be marked as kernel
void markKernel(llvm::Module &m, llvm::Function *kernel) {
  LLVMContext &ctx = m.getContext();
  NamedMDNode *kernelMD = m.getNamedMetadata("nvvm.annotations");

  if (kernelMD) {
    MDNode *node = MDNode::get(
        ctx, {ValueAsMetadata::get(kernel), MDString::get(ctx, "kernel"),
              ValueAsMetadata::get(
                  ConstantInt::get(Type::getInt32Ty(ctx), 1, true))});

    kernelMD->addOperand(node);
  } else {
    errs() << "No kernel meta data found!\n";
  }
}
} // namespace mekong
