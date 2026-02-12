#pragma once

#include "tensor.hpp"

namespace mtf {
namespace core {
namespace ops {

Tensor add(const Tensor& a, const Tensor& b);
Tensor sub(const Tensor& a, const Tensor& b);
Tensor mul(const Tensor& a, const Tensor& b);
Tensor div(const Tensor& a, const Tensor& b);

Tensor add_scalar(const Tensor& a, float scalar);
Tensor mul_scalar(const Tensor& a, float scalar);

Tensor matmul(const Tensor& a, const Tensor& b);
Tensor transpose(const Tensor& a);

Tensor sum(const Tensor& a);
Tensor mean(const Tensor& a);

Tensor relu(const Tensor& a);
Tensor sigmoid(const Tensor& a);
Tensor exp(const Tensor& a);
Tensor log(const Tensor& a);
Tensor max(const Tensor& a, const Tensor& b);

} // namespace ops
} // namespace core
} // namespace mtf
