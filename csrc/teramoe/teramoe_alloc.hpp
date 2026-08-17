#pragma once

#include <cstddef>

// Paddle-backed replacement for `c10::cuda::CUDACachingAllocator::raw_alloc/raw_delete`.
//
// Implemented in `csrc/moe_extension.cpp` (host, gcc-compiled) so that Paddle's
// allocator headers stay out of the nvcc translation units.
namespace teramoe_alloc {

void* raw_alloc(size_t nbytes);
void raw_delete(void* ptr);

}  // namespace teramoe_alloc
