#include "optim/sgd.hpp"

namespace mtf {
namespace optim {

SGD::SGD(std::vector<autograd::NodePtr> parameters, float learning_rate)
    : Optimizer(std::move(parameters)), learning_rate_(learning_rate) {}

void SGD::step() {
    for (auto& param : parameters_) {
        float* p_data = param->value.data();
        const float* g_data = param->grad.data();
        size_t size = param->value.size();

        for (size_t i = 0; i < size; ++i) {
            p_data[i] -= learning_rate_ * g_data[i];
        }
    }
}

} // namespace optim
} // namespace mtf
