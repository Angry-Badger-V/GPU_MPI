cmake_minimum_required(VERSION 3.16)
project(mpi_cuda_coursework)

enable_language(CXX)
enable_language(CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED)
find_package(CUDAToolkit REQUIRED)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  add_compile_options(-Wall -Wextra)
endif()

set(SRC_COMMON
  src/mpi_distribution.cpp
  src/cpu_reference.cpp
  src/cuda_kernels.cu
  src/benchmarks.cpp
)

# ── Main executable ────────────────────────────────────────────────────────────
add_executable(mpi_cuda_coursework
  src/main.cpp
  ${SRC_COMMON}
)

target_include_directories(mpi_cuda_coursework PRIVATE include)

if(TARGET MPI::MPI_CXX)
  target_link_libraries(mpi_cuda_coursework PRIVATE MPI::MPI_CXX)
else()
  target_include_directories(mpi_cuda_coursework PRIVATE ${MPI_CXX_INCLUDE_PATH})
  target_link_libraries(mpi_cuda_coursework PRIVATE ${MPI_CXX_LIBRARIES})
endif()

target_link_libraries(mpi_cuda_coursework PRIVATE CUDA::cudart)

set_target_properties(mpi_cuda_coursework PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  CUDA_ARCHITECTURES "80"          # Adjust to your GPU (e.g. 70=V100, 80=A100)
)

target_compile_options(mpi_cuda_coursework PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
)

# ── Test executable ────────────────────────────────────────────────────────────
add_executable(mpi_cuda_tests
  tests/test_main.cpp
  ${SRC_COMMON}
)

target_include_directories(mpi_cuda_tests PRIVATE include)

if(TARGET MPI::MPI_CXX)
  target_link_libraries(mpi_cuda_tests PRIVATE MPI::MPI_CXX)
else()
  target_include_directories(mpi_cuda_tests PRIVATE ${MPI_CXX_INCLUDE_PATH})
  target_link_libraries(mpi_cuda_tests PRIVATE ${MPI_CXX_LIBRARIES})
endif()

target_link_libraries(mpi_cuda_tests PRIVATE CUDA::cudart)

set_target_properties(mpi_cuda_tests PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  CUDA_ARCHITECTURES "80"
)
