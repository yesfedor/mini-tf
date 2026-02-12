#pragma once

#include "autograd/node.hpp"

namespace mtf {
namespace nn {

class MSELoss {
public:
    autograd::NodePtr operator()(autograd::NodePtr prediction, autograd::NodePtr target);
};

class CrossEntropyLoss {
public:
    autograd::NodePtr operator()(autograd::NodePtr prediction, autograd::NodePtr target);
};

} // namespace nn
} // namespace mtf
