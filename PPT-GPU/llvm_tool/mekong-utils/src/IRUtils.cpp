#include "IRUtils.h"
#include "x65599.h"

#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/Support/raw_ostream.h"

#include <fstream>
#include <sstream>

using namespace llvm;

namespace mekong {

///===------------------------------------------------------------------===//
///                           Utility  Functions
///===------------------------------------------------------------------===//

std::string getModulePrefix(llvm::Module *m) {
  const std::string sourceFile = m->getSourceFileName();

  // Get BaseName
  std::string baseName = "";
  size_t pos = sourceFile.rfind("/");
  if (pos != std::string::npos)
    baseName = sourceFile.substr(pos + 1);
  else
    baseName = sourceFile;

  std::ifstream file(sourceFile, std::ios::binary | std::ios::ate);
  auto size = file.tellg();
  file.seekg(0, std::ios::beg);

  std::string buffer(size, '\0');
  assert(file.read(&buffer[0], size && "Could not read file!"));

  auto hash = generateHash(buffer.c_str(), size);

  std::ostringstream oss;
  oss << baseName << "_" << std::hex << hash;
  return std::string(oss.str());
}

std::map<llvm::BasicBlock *, int>
getBlockIDMap(llvm::Function *func, std::vector<mekong::PTXFunction> funcVec,
              std::string originalFunctionName) {
  PTXFunction funcPTX;
  for (int i = 0; i < funcVec.size(); ++i) {
    if (originalFunctionName == funcVec[i].name)
      funcPTX = funcVec[i];
  }

  // Build BB Name -> ID Map
  std::map<StringRef, int> nameMap;
  for (int i = 0; i < funcPTX.bb.size(); ++i) {
    nameMap.insert({funcPTX.bb[i].name, i});
  }

  // Build Final Map
  std::map<llvm::BasicBlock *, int> ptrMap;
  for (llvm::BasicBlock &bb : func->getBasicBlockList()) {
    ptrMap[&bb] = nameMap[bb.getName()];
  }

  return ptrMap;
}

void dumpModuleToFile(llvm::Module &m, std::string filepath) {

  std::error_code EC;
  { // Seperate name space to prevent usage of file later in code
    // also flushes input of file to disk - very important!
    raw_fd_ostream file(filepath, EC, llvm::sys::fs::OpenFlags(1)); // OF_Text
    // WriteBitcodeToFile()
    assert(!EC);
    llvm::WriteBitcodeToFile(m, file);
    file.flush();
  }
}

void dumpModuleToFile(llvm::Module &m, const char *filepath) {
  dumpModuleToFile(m, std::string(filepath));
}

///===------------------------------------------------------------------===//
///                           Analysis Functions
///===------------------------------------------------------------------===//

///===------------------------------------------------------------------===//
///                        Transformation Functions
///===------------------------------------------------------------------===//

llvm::GlobalVariable *createGlobalString(llvm::Module &M, llvm::StringRef Str,
                                         const llvm::Twine &Name,
                                         unsigned AddressSpace) {
  Constant *StrConstant = ConstantDataArray::getString(M.getContext(), Str);
  auto *GV = new GlobalVariable(
      M, StrConstant->getType(), true, GlobalValue::PrivateLinkage, StrConstant,
      Name, nullptr, GlobalVariable::NotThreadLocal, AddressSpace);
  GV->setUnnamedAddr(GlobalValue::UnnamedAddr::Global);
  // GV->setAlignment(1);
  GV->setAlignment(MaybeAlign(1));

  return GV;
}

llvm::Constant *createGlobalStringPtr(llvm::Module &M, llvm::StringRef Str,
                                      const llvm::Twine &Name,
                                      unsigned AddressSpace) {

  GlobalVariable *GV = createGlobalString(M, Str, Name, AddressSpace);
  Constant *Zero = ConstantInt::get(Type::getInt32Ty(M.getContext()), 0);
  Constant *Indices[] = {Zero, Zero};
  return ConstantExpr::getInBoundsGetElementPtr(GV->getValueType(), GV,
                                                Indices);
}

void linkIR(llvm::StringRef ir, llvm::Module &m) {
  llvm::LLVMContext &ctx = m.getContext();
  llvm::Linker linker(m);
  llvm::SMDiagnostic err;
  std::unique_ptr<llvm::Module> irModule = llvm::parseIR(
      llvm::MemoryBufferRef(*llvm::MemoryBuffer::getMemBuffer(ir, "", true)),
      err, ctx);

  if (!irModule)
    err.print("linkIR", llvm::errs());

  linker.linkInModule(std::move(irModule));
}

llvm::Function *getPrintfFunc(llvm::Module &module) {
  auto &ctx = module.getContext();
  // llvm::FunctionType *printf_type = llvm::TypeBuilder<int(char *, ...),
  // false>::get(module.getContext());
  std::vector<Type *> args = {Type::getInt8PtrTy(ctx)};
  FunctionType *printf_type =
      FunctionType::get(Type::getInt32Ty(ctx), args, true);
  Function *func = llvm::cast<Function>(
      module.getOrInsertFunction("printf", printf_type).getCallee());

  return func;
}

llvm::CallInst *callPrintf(llvm::Module &module, std::string str,
                           llvm::Value *val) {
  llvm::LLVMContext &ctx = module.getContext();
  llvm::ArrayType *type =
      llvm::ArrayType::get(llvm::IntegerType::get(ctx, 8), str.size() + 1);
  llvm::Constant *strConstant = llvm::ConstantDataArray::getString(ctx, str);
  llvm::GlobalVariable *GVStr = new llvm::GlobalVariable(
      module, type, true, llvm::GlobalValue::InternalLinkage, strConstant);
  llvm::Constant *zero =
      llvm::Constant::getNullValue(llvm::IntegerType::getInt32Ty(ctx));

  llvm::IRBuilder<> builder(module.getContext());
  llvm::Value *strVal = builder.CreateGEP(GVStr, {zero, zero});

  std::vector<llvm::Value *> args;
  args.push_back(&*strVal);
  if (val != nullptr)
    args.push_back(val);

  llvm::Function *printf_func = getPrintfFunc(module);
  printf_func->setCallingConv(llvm::CallingConv::C);
  llvm::CallInst *ci = builder.CreateCall(printf_func, args);
  return ci;
}

llvm::CallInst *callPrintf(llvm::Module &module, llvm::IRBuilder<> &builder,
                           std::string str, llvm::Value *val) {
  LLVMContext &ctx = builder.getContext();
  GlobalVariable *stringVal = builder.CreateGlobalString(str);

  Constant *zero = builder.getInt32(0);

  Value *strVal = builder.CreateGEP(stringVal, {zero, zero});

  std::vector<llvm::Value *> args;
  args.push_back(&*strVal);
  if (val != nullptr)
    args.push_back(val);

  Function *printf_func = getPrintfFunc(module);
  printf_func->setCallingConv(llvm::CallingConv::C);
  CallInst *ci = builder.CreateCall(printf_func, args);
  return ci;
}

} // namespace mekong
