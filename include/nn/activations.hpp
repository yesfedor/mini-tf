#pragma once

#include "autograd/node.hpp"

namespace mtf {
namespace nn {
namespace functional {

autograd::NodePtr relu(autograd::NodePtr input);
autograd::NodePtr sigmoid(autograd::NodePtr input);
autograd::NodePtr softmax(autograd::NodePtr input);

} // namespace functional
} // namespace nn
} // namespace mtf
