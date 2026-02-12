#include "optim/optimizer.hpp"

namespace mtf {
namespace optim {

Optimizer::Optimizer(std::vector<autograd::NodePtr> parameters) 
    : parameters_(std::move(parameters)) {}

void Optimizer::zero_grad() {
    for (auto& param : parameters_) {
        param->zero_grad();
    }
}

} // namespace optim
} // namespace mtf
