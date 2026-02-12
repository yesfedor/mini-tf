#pragma once

#include <vector>
#include "autograd/node.hpp"

namespace mtf {
namespace optim {

class Optimizer {
public:
    Optimizer(std::vector<autograd::NodePtr> parameters);
    virtual ~Optimizer() = default;

    void zero_grad();
    virtual void step() = 0;

protected:
    std::vector<autograd::NodePtr> parameters_;
};

} // namespace optim
} // namespace mtf
