#ifndef BASICBLOCKENUMERATOR_H
#define BASICBLOCKENUMERATOR_H

#include "llvm/IR/BasicBlock.h"

namespace mekong {
void visitNodes(llvm::BasicBlock *begin, llvm::BasicBlock *end, void *data,
                bool (*visitorFunc)(llvm::BasicBlock *, void *));
}

#endif
