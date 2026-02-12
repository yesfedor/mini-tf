#pragma once

#include "optimizer.hpp"

namespace mtf {
namespace optim {

class SGD : public Optimizer {
public:
    SGD(std::vector<autograd::NodePtr> parameters, float learning_rate = 0.01f);

    void step() override;

private:
    float learning_rate_;
};

} // namespace optim
} // namespace mtf
