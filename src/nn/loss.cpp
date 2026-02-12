#include "nn/loss.hpp"
#include "core/ops_cpu.hpp"

namespace mtf {
namespace nn {

autograd::NodePtr MSELoss::operator()(autograd::NodePtr prediction, autograd::NodePtr target) {
    auto diff = prediction - target;
    auto sq_diff = diff * diff;
    
    core::Tensor val = core::ops::mean(sq_diff->value);
    auto result = autograd::Node::create(val, true, "MSELoss");
    result->parents = {prediction, target};
    
    result->backward_fn = [result, prediction, target]() {
        size_t N = prediction->value.size();
        float scale = 2.0f / static_cast<float>(N);
        
        if (prediction->requires_grad) {
            float grad_loss = result->grad[0];
            
            auto p_data = prediction->value.data();
            auto t_data = target->value.data();
            auto p_grad = prediction->grad.data();
            
            for (size_t i = 0; i < N; ++i) {
                p_grad[i] += grad_loss * scale * (p_data[i] - t_data[i]);
            }
        }
    };
    
    return result;
}

autograd::NodePtr CrossEntropyLoss::operator()(autograd::NodePtr prediction, autograd::NodePtr target) {
    core::Tensor p_val = prediction->value;
    core::Tensor t_val = target->value;
    
    size_t N = p_val.size();
    size_t batch_size = p_val.shape()[0];
    
    float loss_sum = 0.0f;
    for (size_t i = 0; i < N; ++i) {
        float p = std::max(p_val[i], 1e-7f);
        loss_sum -= t_val[i] * std::log(p);
    }
    float loss_mean = loss_sum / static_cast<float>(batch_size);
    
    auto result = autograd::Node::create(core::Tensor(core::Tensor::Shape{1}, {loss_mean}), true, "CELoss");
    result->parents = {prediction};
    
    result->backward_fn = [result, prediction, target, batch_size, N]() {
        if (prediction->requires_grad) {
            float grad_loss = result->grad[0];
            float scale = 1.0f / static_cast<float>(batch_size);
            
            auto p_data = prediction->value.data();
            auto t_data = target->value.data();
            auto p_grad = prediction->grad.data();
            
            for (size_t i = 0; i < N; ++i) {
                float p = std::max(p_data[i], 1e-7f);
                p_grad[i] += grad_loss * scale * (-t_data[i] / p);
            }
        }
    };
    
    return result;
}

} // namespace nn
} // namespace mtf
