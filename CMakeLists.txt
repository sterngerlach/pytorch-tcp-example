
# CMakeLists.txt

cmake_minimum_required(VERSION 3.16)

project(pytorch_model_transfer CXX)

# Find Boost
set(BOOST_USE_STATIC_LIBS ON)
find_package(Boost REQUIRED)

# Path to the LibTorch
set(TORCH_ROOT_DIR $ENV{HOME}/libtorch-1.11.0)
# Find LibTorch
find_package(Torch REQUIRED PATHS ${TORCH_ROOT_DIR})

message(STATUS "LibTorch include directories: ${TORCH_INCLUDE_DIRS}")
message(STATUS "LibTorch libraries: ${TORCH_LIBRARIES}")
message(STATUS "LibTorch compiler flags: ${TORCH_CXX_FLAGS}")

set(CMAKE_DEBUG_POSTFIX "d")

add_executable(client ${CMAKE_CURRENT_SOURCE_DIR}/client.cpp)
target_include_directories(client PUBLIC ${TORCH_INCLUDE_DIRS})
target_link_libraries(client PUBLIC ${TORCH_LIBRARIES})
target_compile_options(client PUBLIC ${TORCH_CXX_FLAGS}
  $<$<CXX_COMPILER_ID:GNU>:-pedantic -Wall -Wextra -Wno-psabi>
  $<$<CONFIG:Release>:-O3>
  $<$<AND:$<CXX_COMPILER_ID:GNU>,$<CONFIG:Debug>>:-O0 -ggdb>)
target_compile_features(client PUBLIC cxx_std_17)
set_target_properties(client PROPERTIES DEBUG_POSTFIX ${CMAKE_DEBUG_POSTFIX})
