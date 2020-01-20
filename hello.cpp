#include <fstream>   // file io
#include <libxsmm.h> // xsmm
#include <iostream>  // cout
#include <vector>    // vector
#include <cstdlib>   // rand srand
#include <ctime>     // time

int main(int argc, char* argv[])
{
  //
  typedef float value_type;
  int batchsize = 1, m = 32, n = 32, k = 32;
  std::vector<value_type> a(batchsize*m*k), b(batchsize*k*n), c(m*n, 0);

  /* C/C++ and Fortran interfaces are available */
  typedef libxsmm_mmfunction<value_type> kernel_type;
  /* generates and dispatches a matrix multiplication kernel (C++ functor) */
  kernel_type kernel(LIBXSMM_GEMM_FLAG_NONE, m,n,k, 1.0/*alpha*/, 0.0/*beta*/);
  assert(kernel);

  // prepare random init
  srand(static_cast<unsigned> (time(0)));

  // fill the input matrices A and B
  for (int i = 0; i < batchsize; ++i) { /* initialize input */
      for(int ia = 0 ; ia < m*k ; ++ia) a[ia] = (static_cast <float> (rand()) / (static_cast <float> (RAND_MAX/2))) - 1.0f; 
      for(int ib = 0 ; ib < k*n ; ++ib) b[ib] = (static_cast <float> (rand()) / (static_cast <float> (RAND_MAX/2))) - 1.0f; 
  }

  // print values of input matrices to "verify" correctness. uncomment
  // for (auto const& v: a) std::cout << v << " ";
  // std::cout << "\n\n" << std::endl;
  // for (auto const& v: b) std::cout << v << " ";
  // std::cout << "\n\n" << std::endl;

  // save the matrice into MatA.raw & MatB.raw
  std::ofstream MatAFile("MatA.raw", std::ofstream::out | std::ofstream::binary);
  std::ofstream MatBFile("MatB.raw", std::ofstream::out | std::ofstream::binary);
  for (auto const& v: a) MatAFile.write((char *) &v, sizeof(value_type));
  for (auto const& v: b) MatBFile.write((char *) &v, sizeof(value_type));
  MatAFile.close();
  MatBFile.close();

  system("python3 posit_MMM.py");

  /* kernel multiplies and accumulates matrix products: C += Ai * Bi */
  for (int i = 0; i < batchsize; ++i) kernel(&a[i*m*k], &b[i*k*n], &c[0]);

  // for (auto const& v: c) std::cout << v << " ";
}
