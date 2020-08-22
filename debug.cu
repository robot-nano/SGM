//
// Created by wserver on 2020/7/23.
//

#include "debug.h"

void debug_log(const char* str) {
#if LOG
  std::cout << str << std::endl;
#endif
}