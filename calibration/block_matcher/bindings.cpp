#include "block_matcher.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <pybind11/numpy.h>
#include <pybind11/iostream.h>


namespace py = pybind11;


PYBIND11_MODULE(eigen_sgbm, m)
{
    m.doc() = "Attempt at custom block matcher";
    m.attr("__version__") = "0.0.1";

}
