#pragma once

#include <Eigen/Dense>


using namespace Eigen;

template<class Derived>
void block_match(const MatrixBase<Derived>& left, const MatrixBase<Derived>& right);