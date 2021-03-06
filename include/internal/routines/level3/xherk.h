
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file implements the Xherk routine. The precision is implemented using the template argument
// 'T', whereas the alpha/beta arguments are of type 'U'. The implementation is very similar to the
// Xsyrk routine.
//
// =================================================================================================

#ifndef CLBLAST_ROUTINES_XHERK_H_
#define CLBLAST_ROUTINES_XHERK_H_

#include "internal/routine.h"

namespace clblast {
// =================================================================================================

// See comment at top of file for a description of the class
template <typename T, typename U>
class Xherk: public Routine<T> {
 public:

  // Members and methods from the base class
  using Routine<T>::db_;
  using Routine<T>::source_string_;
  using Routine<T>::queue_;
  using Routine<T>::context_;
  using Routine<T>::GetProgramFromCache;
  using Routine<T>::PadCopyTransposeMatrix;
  using Routine<T>::TestMatrixA;
  using Routine<T>::TestMatrixC;
  using Routine<T>::RunKernel;
  using Routine<T>::ErrorIn;

  // Constructor
  Xherk(Queue &queue, Event &event, const std::string &name = "HERK");

  // Templated-precision implementation of the routine
  StatusCode DoHerk(const Layout layout, const Triangle triangle, const Transpose a_transpose,
                    const size_t n, const size_t k,
                    const U alpha,
                    const Buffer<T> &a_buffer, const size_t a_offset, const size_t a_ld,
                    const U beta,
                    const Buffer<T> &c_buffer, const size_t c_offset, const size_t c_ld);

 private:
  // Static variable to get the precision
  const static Precision precision_;
};

// =================================================================================================
} // namespace clblast

// CLBLAST_ROUTINES_XHERK_H_
#endif
