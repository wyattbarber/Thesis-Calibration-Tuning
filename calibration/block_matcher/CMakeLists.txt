cmake_minimum_required(VERSION 3.0)
project(eigen_sgbm VERSION 0.0.1)

find_package(Eigen3 REQUIRED HINTS "C:\\Users\\wyatt\\Documents\\vcpkg\\installed\\x64-windows\\share\\eigen3")

# Install python module
find_package(Python3 3.11 COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(eigen_sgbm bindings.cpp)
target_link_libraries(eigen_sgbm PRIVATE Eigen3::Eigen)

message(STATUS "Found Python3 installation location: ${Python3_SITEARCH}")
install(TARGETS eigen_sgbm
    COMPONENT python
    LIBRARY DESTINATION "${Python3_SITEARCH}"
    ARCHIVE DESTINATION "lib"
    RUNTIME DESTINATION "bin")