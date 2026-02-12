#pragma once

#include <cstdlib>
#include <memory>
#include <vector>

namespace mtf {
namespace core {

void* aligned_alloc(size_t size, size_t alignment = 64);
void aligned_free(void* ptr);

} // namespace core
} // namespace mtf
