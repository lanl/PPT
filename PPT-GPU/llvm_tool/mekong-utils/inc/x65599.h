#ifndef X65599_H
#define X65599_H

// A hash function with multiplier 65599 (from Red Dragon book)
unsigned int generateHash(const char *string, size_t len) {
  unsigned int hash = 0;
  for (size_t i = 0; i < len; ++i) {
    hash = 65599 * hash + string[i];
  }
  return hash ^ (hash >> 16);
}

#endif
