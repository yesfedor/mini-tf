#include "optim/adam.hpp"
#include <cmath>

namespace mtf {
namespace optim {

Adam::Adam(std::vector<autograd::NodePtr> parameters, 
           float lr, float beta1, float beta2, float epsilon)
    : Optimizer(std::move(parameters)), 
      lr_(lr), beta1_(beta1), beta2_(beta2), epsilon_(epsilon), t_(0) {
    
    for (const auto& param : parameters_) {
        core::Tensor m(param->value.shape());
        m.fill(0.0f);
        m_.push_back(std::move(m));

        core::Tensor v(param->value.shape());
        v.fill(0.0f);
        v_.push_back(std::move(v));
    }
}

void Adam::step() {
    t_++;
    
    float bias_correction1 = 1.0f - static_cast<float>(std::pow(beta1_, t_));
    float bias_correction2 = 1.0f - static_cast<float>(std::pow(beta2_, t_));

    for (size_t i = 0; i < parameters_.size(); ++i) {
        auto& param = parameters_[i];
        auto& m = m_[i];
        auto& v = v_[i];

        float* p_data = param->value.data();
        const float* g_data = param->grad.data();
        float* m_data = m.data();
        float* v_data = v.data();
        size_t size = param->value.size();

        for (size_t j = 0; j < size; ++j) {
            float g = g_data[j];

            m_data[j] = beta1_ * m_data[j] + (1.0f - beta1_) * g;
            v_data[j] = beta2_ * v_data[j] + (1.0f - beta2_) * g * g;

            float m_hat = m_data[j] / bias_correction1;
            float v_hat = v_data[j] / bias_correction2;

            p_data[j] -= lr_ * m_hat / (std::sqrt(v_hat) + epsilon_);
        }
    }
}

} // namespace optim
} // namespace mtf
