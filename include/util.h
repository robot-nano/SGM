//
// Created by wserver on 2020/5/31.
//

#ifndef SGM_INCLUDE_UTIL_H_
#define SGM_INCLUDE_UTIL_H_

#include <cstdio>

#define DEBUG_LOG(x)           \
  do {                         \
  fprintf(stderr, "%s", (x));  \
} while (0)

#define SGM_ABORT abort()

#ifdef NDEBUG
#define SGM_ASSERT_FALSE (static_cast<void>(0))
#else
#define SGM_ASSERT_FALSE SGM_ABORT
#endif

#define SGM_FATAL(msg)         \
  do {                         \
  DEBUG_LOG(msg);              \
  DEBUG_LOG("\nFATAL\n");      \
  SGM_ABORT;                   \
} while (0)

#define SGM_ASSERT(x)                \
  do {                               \
    if (!(x) SGM_FATAL(#x));         \
} while (0)


#ifndef SGM_CHECK_EQ
#define SGM_CHECK_EQ(x, y) ((x) == (y)) ? (void)0 : SGM_ASSERT_FALSE;
#endif

#endif //SGM_INCLUDE_UTIL_H_
