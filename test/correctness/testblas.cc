
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file implements the TestBlas class (see the header for information about the class).
//
// =================================================================================================

#include <algorithm>

#include "correctness/testblas.h"

namespace clblast {
// =================================================================================================

// The transpose-options to test with (data-type dependent)
template <> const std::vector<Transpose> TestBlas<float,float>::kTransposes = {Transpose::kNo, Transpose::kYes};
template <> const std::vector<Transpose> TestBlas<double,double>::kTransposes = {Transpose::kNo, Transpose::kYes};
template <> const std::vector<Transpose> TestBlas<float2,float2>::kTransposes = {Transpose::kNo, Transpose::kYes, Transpose::kConjugate};
template <> const std::vector<Transpose> TestBlas<double2,double2>::kTransposes = {Transpose::kNo, Transpose::kYes, Transpose::kConjugate};
template <> const std::vector<Transpose> TestBlas<float2,float>::kTransposes = {Transpose::kNo, Transpose::kConjugate};
template <> const std::vector<Transpose> TestBlas<double2,double>::kTransposes = {Transpose::kNo, Transpose::kConjugate};

// =================================================================================================

// Constructor, initializes the base class tester and input data
template <typename T, typename U>
TestBlas<T,U>::TestBlas(int argc, char *argv[], const bool silent,
                        const std::string &name, const std::vector<std::string> &options,
                        const Routine run_routine, const Routine run_reference,
                        const ResultGet get_result, const ResultIndex get_index,
                        const ResultIterator get_id1, const ResultIterator get_id2):
    Tester<T,U>(argc, argv, silent, name, options),
    run_routine_(run_routine),
    run_reference_(run_reference),
    get_result_(get_result),
    get_index_(get_index),
    get_id1_(get_id1),
    get_id2_(get_id2) {

  // Computes the maximum sizes. This allows for a single set of input/output buffers.
  auto max_vec = *std::max_element(kVectorDims.begin(), kVectorDims.end());
  auto max_inc = *std::max_element(kIncrements.begin(), kIncrements.end());
  auto max_mat = *std::max_element(kMatrixDims.begin(), kMatrixDims.end());
  auto max_ld = *std::max_element(kMatrixDims.begin(), kMatrixDims.end());
  auto max_matvec = *std::max_element(kMatrixVectorDims.begin(), kMatrixVectorDims.end());
  auto max_offset = *std::max_element(kOffsets.begin(), kOffsets.end());

  // Creates test input data
  x_source_.resize(std::max(max_vec, max_matvec)*max_inc + max_offset);
  y_source_.resize(std::max(max_vec, max_matvec)*max_inc + max_offset);
  a_source_.resize(std::max(max_mat, max_matvec)*std::max(max_ld, max_matvec) + max_offset);
  b_source_.resize(std::max(max_mat, max_matvec)*std::max(max_ld, max_matvec) + max_offset);
  c_source_.resize(std::max(max_mat, max_matvec)*std::max(max_ld, max_matvec) + max_offset);
  ap_source_.resize(std::max(max_mat, max_matvec)*std::max(max_mat, max_matvec) + max_offset);
  dot_source_.resize(std::max(max_mat, max_matvec) + max_offset);
  PopulateVector(x_source_);
  PopulateVector(y_source_);
  PopulateVector(a_source_);
  PopulateVector(b_source_);
  PopulateVector(c_source_);
  PopulateVector(ap_source_);
  PopulateVector(dot_source_);
}

// ===============================================================================================

// Tests the routine for a wide variety of parameters
template <typename T, typename U>
void TestBlas<T,U>::TestRegular(std::vector<Arguments<U>> &test_vector, const std::string &name) {
  if (!PrecisionSupported<T>(device_)) { return; }
  TestStart("regular behaviour", name);

  // Iterates over all the to-be-tested combinations of arguments
  for (auto &args: test_vector) {

    // Runs the reference clBLAS code
    auto x_vec1 = Buffer<T>(context_, args.x_size);
    auto y_vec1 = Buffer<T>(context_, args.y_size);
    auto a_mat1 = Buffer<T>(context_, args.a_size);
    auto b_mat1 = Buffer<T>(context_, args.b_size);
    auto c_mat1 = Buffer<T>(context_, args.c_size);
    auto ap_mat1 = Buffer<T>(context_, args.ap_size);
    auto dot1 = Buffer<T>(context_, args.dot_size);
    x_vec1.Write(queue_, args.x_size, x_source_);
    y_vec1.Write(queue_, args.y_size, y_source_);
    a_mat1.Write(queue_, args.a_size, a_source_);
    b_mat1.Write(queue_, args.b_size, b_source_);
    c_mat1.Write(queue_, args.c_size, c_source_);
    ap_mat1.Write(queue_, args.ap_size, ap_source_);
    dot1.Write(queue_, args.dot_size, dot_source_);
    auto buffers1 = Buffers<T>{x_vec1, y_vec1, a_mat1, b_mat1, c_mat1, ap_mat1, dot1};
    auto status1 = run_reference_(args, buffers1, queue_);

    // Runs the CLBlast code
    auto x_vec2 = Buffer<T>(context_, args.x_size);
    auto y_vec2 = Buffer<T>(context_, args.y_size);
    auto a_mat2 = Buffer<T>(context_, args.a_size);
    auto b_mat2 = Buffer<T>(context_, args.b_size);
    auto c_mat2 = Buffer<T>(context_, args.c_size);
    auto ap_mat2 = Buffer<T>(context_, args.ap_size);
    auto dot2 = Buffer<T>(context_, args.dot_size);
    x_vec2.Write(queue_, args.x_size, x_source_);
    y_vec2.Write(queue_, args.y_size, y_source_);
    a_mat2.Write(queue_, args.a_size, a_source_);
    b_mat2.Write(queue_, args.b_size, b_source_);
    c_mat2.Write(queue_, args.c_size, c_source_);
    ap_mat2.Write(queue_, args.ap_size, ap_source_);
    dot2.Write(queue_, args.dot_size, dot_source_);
    auto buffers2 = Buffers<T>{x_vec2, y_vec2, a_mat2, b_mat2, c_mat2, ap_mat2, dot2};
    auto status2 = run_routine_(args, buffers2, queue_);

    // Tests for equality of the two status codes
    if (status1 != StatusCode::kSuccess || status2 != StatusCode::kSuccess) {
      TestErrorCodes(status1, status2, args);
      continue;
    }

    // Downloads the results
    auto result1 = get_result_(args, buffers1, queue_);
    auto result2 = get_result_(args, buffers2, queue_);

    // Checks for differences in the output
    auto errors = size_t{0};
    for (auto id1=size_t{0}; id1<get_id1_(args); ++id1) {
      for (auto id2=size_t{0}; id2<get_id2_(args); ++id2) {
        auto index = get_index_(args, id1, id2);
        if (!TestSimilarity(result1[index], result2[index])) {
          errors++;
        }
      }
    }

    // Tests the error count (should be zero)
    TestErrorCount(errors, get_id1_(args)*get_id2_(args), args);
  }
  TestEnd();
}

// =================================================================================================

// Tests the routine for cases with invalid OpenCL memory buffer sizes. Tests only on return-types,
// does not test for results (if any).
template <typename T, typename U>
void TestBlas<T,U>::TestInvalid(std::vector<Arguments<U>> &test_vector, const std::string &name) {
  if (!PrecisionSupported<T>(device_)) { return; }
  TestStart("invalid buffer sizes", name);

  // Iterates over all the to-be-tested combinations of arguments
  for (auto &args: test_vector) {

    // Creates the OpenCL buffers. Note: we are not using the C++ version since we explicitly
    // want to be able to create invalid buffers (no error checking here).
    auto x1 = clCreateBuffer(context_(), CL_MEM_READ_WRITE, args.x_size*sizeof(T), nullptr,nullptr);
    auto y1 = clCreateBuffer(context_(), CL_MEM_READ_WRITE, args.y_size*sizeof(T), nullptr,nullptr);
    auto a1 = clCreateBuffer(context_(), CL_MEM_READ_WRITE, args.a_size*sizeof(T), nullptr,nullptr);
    auto b1 = clCreateBuffer(context_(), CL_MEM_READ_WRITE, args.b_size*sizeof(T), nullptr,nullptr);
    auto c1 = clCreateBuffer(context_(), CL_MEM_READ_WRITE, args.c_size*sizeof(T), nullptr,nullptr);
    auto ap1 = clCreateBuffer(context_(), CL_MEM_READ_WRITE, args.ap_size*sizeof(T), nullptr,nullptr);
    auto d1 = clCreateBuffer(context_(), CL_MEM_READ_WRITE, args.dot_size*sizeof(T), nullptr,nullptr);
    auto x_vec1 = Buffer<T>(x1);
    auto y_vec1 = Buffer<T>(y1);
    auto a_mat1 = Buffer<T>(a1);
    auto b_mat1 = Buffer<T>(b1);
    auto c_mat1 = Buffer<T>(c1);
    auto ap_mat1 = Buffer<T>(ap1);
    auto dot1 = Buffer<T>(d1);
    auto x2 = clCreateBuffer(context_(), CL_MEM_READ_WRITE, args.x_size*sizeof(T), nullptr,nullptr);
    auto y2 = clCreateBuffer(context_(), CL_MEM_READ_WRITE, args.y_size*sizeof(T), nullptr,nullptr);
    auto a2 = clCreateBuffer(context_(), CL_MEM_READ_WRITE, args.a_size*sizeof(T), nullptr,nullptr);
    auto b2 = clCreateBuffer(context_(), CL_MEM_READ_WRITE, args.b_size*sizeof(T), nullptr,nullptr);
    auto c2 = clCreateBuffer(context_(), CL_MEM_READ_WRITE, args.c_size*sizeof(T), nullptr,nullptr);
    auto ap2 = clCreateBuffer(context_(), CL_MEM_READ_WRITE, args.ap_size*sizeof(T), nullptr,nullptr);
    auto d2 = clCreateBuffer(context_(), CL_MEM_READ_WRITE, args.dot_size*sizeof(T), nullptr,nullptr);
    auto x_vec2 = Buffer<T>(x2);
    auto y_vec2 = Buffer<T>(y2);
    auto a_mat2 = Buffer<T>(a2);
    auto b_mat2 = Buffer<T>(b2);
    auto c_mat2 = Buffer<T>(c2);
    auto ap_mat2 = Buffer<T>(ap2);
    auto dot2 = Buffer<T>(d2);

    // Runs the two routines
    auto buffers1 = Buffers<T>{x_vec1, y_vec1, a_mat1, b_mat1, c_mat1, ap_mat1, dot1};
    auto buffers2 = Buffers<T>{x_vec2, y_vec2, a_mat2, b_mat2, c_mat2, ap_mat2, dot2};
    auto status1 = run_reference_(args, buffers1, queue_);
    auto status2 = run_routine_(args, buffers2, queue_);

    // Tests for equality of the two status codes
    TestErrorCodes(status1, status2, args);
  }
  TestEnd();
}

// =================================================================================================

// Compiles the templated class
template class TestBlas<float, float>;
template class TestBlas<double, double>;
template class TestBlas<float2, float2>;
template class TestBlas<double2, double2>;
template class TestBlas<float2, float>;
template class TestBlas<double2, double>;

// =================================================================================================
} // namespace clblast
