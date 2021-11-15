#ifndef PTXFUNCTION_H
#define PTXFUNCTION_H

#include <string>
#include <vector>

namespace mekong {
// { FunctionName, { BasicBlock : InstList }}
struct PTXFunction {
  std::string name;
  struct Block {
    std::string name;
    std::vector<std::string> inst;
  };

  std::vector<Block> bb;
};
} // namespace mekong

#endif
