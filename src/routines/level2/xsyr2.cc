
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file implements the Xsyr2 class (see the header for information about the class).
//
// =================================================================================================

#include "internal/routines/level2/xsyr2.h"

#include <string>

namespace clblast {
// =================================================================================================

// Constructor: forwards to base class constructor
template <typename T>
Xsyr2<T>::Xsyr2(Queue &queue, Event &event, const std::string &name):
    Xher2<T>(queue, event, name) {
}

// =================================================================================================

// The main routine
template <typename T>
StatusCode Xsyr2<T>::DoSyr2(const Layout layout, const Triangle triangle,
                            const size_t n,
                            const T alpha,
                            const Buffer<T> &x_buffer, const size_t x_offset, const size_t x_inc,
                            const Buffer<T> &y_buffer, const size_t y_offset, const size_t y_inc,
                            const Buffer<T> &a_buffer, const size_t a_offset, const size_t a_ld) {

  // Specific Xsyr2 functionality is implemented in the kernel using defines
  return DoHer2(layout, triangle, n, alpha,
                x_buffer, x_offset, x_inc,
                y_buffer, y_offset, y_inc,
                a_buffer, a_offset, a_ld);
}

// =================================================================================================

// Compiles the templated class
template class Xsyr2<float>;
template class Xsyr2<double>;

// =================================================================================================
} // namespace clblast
