#include "BasicBlockEnumerator.h"
#include "llvm/IR/Instructions.h"
#include <llvm/Support/raw_ostream.h>

#include <deque>
#include <set>

using namespace llvm;

bool contains(BasicBlock *key, std::deque<BasicBlock *> &frontier) {
  for (BasicBlock *item : frontier) {
    if (item == key)
      return true;
  }
  return false;
}

void expand(BasicBlock *node, std::set<BasicBlock *> &visited,
            std::deque<BasicBlock *> &frontier) {
  Instruction *ti = node->getTerminator();
  if (BranchInst *bi = dyn_cast_or_null<BranchInst>(ti)) {
    for (BasicBlock *successor : bi->successors()) {
      // Add to frontier if not already vsited or not already in list to visit
      if (visited.find(successor) == visited.end() and
          not contains(successor, frontier)) {
        frontier.push_back(successor);
      }
    }
  } else if (InvokeInst *invi = dyn_cast_or_null<InvokeInst>(ti)) {
    for (int i = 0; i < invi->getNumSuccessors(); ++i) {
      BasicBlock *successor = invi->getSuccessor(i);
      // Add to frontier if not already vsited or not already in list to visit
      if (visited.find(successor) == visited.end() and
          not contains(successor, frontier)) {
        frontier.push_back(successor);
      }
    }
  }
}

namespace mekong {
void visitNodes(BasicBlock *begin, BasicBlock *end, void *data,
                bool (*visitorFunc)(BasicBlock *, void *)) {
  std::set<BasicBlock *> visited = {end};
  std::deque<BasicBlock *> frontier = {begin};

  while (frontier.size() > 0) {
    BasicBlock *currentNode = frontier.front();
    frontier.pop_front();

    // Visit
    // If true is returned there is no need to visit further nodes
    if (visitorFunc(currentNode, data))
      return;
    visited.insert(currentNode);

    // Add further nodes to visit
    expand(currentNode, visited, frontier);
  }
}
} // namespace mekong
