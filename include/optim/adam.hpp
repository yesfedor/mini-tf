#pragma once

#include "optimizer.hpp"
#include <map>
#include "core/tensor.hpp"

namespace mtf {
namespace optim {

class Adam : public Optimizer {
public:
    Adam(std::vector<autograd::NodePtr> parameters, 
         float lr = 0.001f, 
         float beta1 = 0.9f, 
         float beta2 = 0.999f, 
         float epsilon = 1e-8f);

    void step() override;

private:
    float lr_;
    float beta1_;
    float beta2_;
    float epsilon_;
    int t_; 

    std::vector<core::Tensor> m_;
    std::vector<core::Tensor> v_;
};

} // namespace optim
} // namespace mtf
