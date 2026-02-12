#include "core/memory.hpp"
#include <cstdlib>

#if defined(_MSC_VER)
#include <malloc.h>
#endif

namespace mtf {
namespace core {

void* aligned_alloc(size_t size, size_t alignment) {
#if defined(_MSC_VER)
    return _aligned_malloc(size, alignment);
#else
    return std::aligned_alloc(alignment, size);
#endif
}

void aligned_free(void* ptr) {
#if defined(_MSC_VER)
    _aligned_free(ptr);
#else
    std::free(ptr);
#endif
}

} // namespace core
} // namespace mtf
