
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file implements the Xhpr class (see the header for information about the class).
//
// =================================================================================================

#include "internal/routines/level2/xhpr.h"

#include <string>

namespace clblast {
// =================================================================================================

// Constructor: forwards to base class constructor
template <typename T, typename U>
Xhpr<T,U>::Xhpr(Queue &queue, Event &event, const std::string &name):
    Xher<T,U>(queue, event, name) {
}

// =================================================================================================

// The main routine
template <typename T, typename U>
StatusCode Xhpr<T,U>::DoHpr(const Layout layout, const Triangle triangle,
                            const size_t n,
                            const U alpha,
                            const Buffer<T> &x_buffer, const size_t x_offset, const size_t x_inc,
                            const Buffer<T> &ap_buffer, const size_t ap_offset) {

  // Specific Xhpr functionality is implemented in the kernel using defines
  return DoHer(layout, triangle, n, alpha,
               x_buffer, x_offset, x_inc,
               ap_buffer, ap_offset, n,
               true); // packed matrix
}

// =================================================================================================

// Compiles the templated class
template class Xhpr<float2, float>;
template class Xhpr<double2, double>;

// =================================================================================================
} // namespace clblast
