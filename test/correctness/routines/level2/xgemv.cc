
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// =================================================================================================

#include "correctness/testblas.h"
#include "routines/level2/xgemv.h"

// Shortcuts to the clblast namespace
using float2 = clblast::float2;
using double2 = clblast::double2;

// Main function (not within the clblast namespace)
int main(int argc, char *argv[]) {
  clblast::RunTests<clblast::TestXgemv<float>, float, float>(argc, argv, false, "SGEMV");
  clblast::RunTests<clblast::TestXgemv<double>, double, double>(argc, argv, true, "DGEMV");
  clblast::RunTests<clblast::TestXgemv<float2>, float2, float2>(argc, argv, true, "CGEMV");
  clblast::RunTests<clblast::TestXgemv<double2>, double2, double2>(argc, argv, true, "ZGEMV");
  return 0;
}

// =================================================================================================
