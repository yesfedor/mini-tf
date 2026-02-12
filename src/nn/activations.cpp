#include "nn/activations.hpp"
#include "core/ops_cpu.hpp"

namespace mtf {
namespace nn {
namespace functional {

autograd::NodePtr relu(autograd::NodePtr input) {
    auto result = autograd::Node::create(core::ops::relu(input->value), 
                                         input->requires_grad, 
                                         "ReLU");
    result->parents = {input};

    result->backward_fn = [result, input]() {
        if (input->requires_grad) {
            const float* in_data = input->value.data();
            const float* grad_out = result->grad.data();
            float* grad_in = input->grad.data();
            size_t size = input->value.size();
            
            for (size_t i = 0; i < size; ++i) {
                if (in_data[i] > 0) {
                    grad_in[i] += grad_out[i];
                }
            }
        }
    };
    return result;
}

autograd::NodePtr sigmoid(autograd::NodePtr input) {
    auto result = autograd::Node::create(core::ops::sigmoid(input->value), 
                                         input->requires_grad, 
                                         "Sigmoid");
    result->parents = {input};

    result->backward_fn = [result, input]() {
        if (input->requires_grad) {
            const float* y_data = result->value.data();
            const float* grad_out = result->grad.data();
            float* grad_in = input->grad.data();
            size_t size = input->value.size();

            for (size_t i = 0; i < size; ++i) {
                float s = y_data[i];
                grad_in[i] += grad_out[i] * s * (1.0f - s);
            }
        }
    };
    return result;
}

autograd::NodePtr tanh(autograd::NodePtr input) {
    auto result = autograd::Node::create(core::ops::tanh(input->value), 
                                         input->requires_grad, 
                                         "Tanh");
    result->parents = {input};

    result->backward_fn = [result, input]() {
        if (input->requires_grad) {
            const float* y_data = result->value.data();
            const float* grad_out = result->grad.data();
            float* grad_in = input->grad.data();
            size_t size = input->value.size();

            for (size_t i = 0; i < size; ++i) {
                float t = y_data[i];
                grad_in[i] += grad_out[i] * (1.0f - t * t);
            }
        }
    };
    return result;
}

autograd::NodePtr softmax(autograd::NodePtr input) {
    size_t rows = input->value.shape()[0];
    size_t cols = input->value.shape()[1];
    
    core::Tensor out_tensor(input->value.shape());
    
    const float* in_ptr = input->value.data();
    float* out_ptr = out_tensor.data();
    
    for (size_t i = 0; i < rows; ++i) {
        float max_val = -1e9;
        for (size_t j = 0; j < cols; ++j) {
            max_val = std::max(max_val, in_ptr[i * cols + j]);
        }
        
        float sum_exp = 0.0f;
        for (size_t j = 0; j < cols; ++j) {
            float val = std::exp(in_ptr[i * cols + j] - max_val);
            out_ptr[i * cols + j] = val;
            sum_exp += val;
        }
        
        for (size_t j = 0; j < cols; ++j) {
            out_ptr[i * cols + j] /= sum_exp;
        }
    }
    
    auto result = autograd::Node::create(out_tensor, input->requires_grad, "Softmax");
    result->parents = {input};
    
    result->backward_fn = [result, input, rows, cols]() {
        if (input->requires_grad) {
            const float* y_ptr = result->value.data();
            const float* dy_ptr = result->grad.data();
            float* dx_ptr = input->grad.data();
            
            for (size_t i = 0; i < rows; ++i) {
                float dot = 0.0f;
                for (size_t j = 0; j < cols; ++j) {
                    dot += y_ptr[i * cols + j] * dy_ptr[i * cols + j];
                }
                
                for (size_t j = 0; j < cols; ++j) {
                    float y = y_ptr[i * cols + j];
                    float dy = dy_ptr[i * cols + j];
                    dx_ptr[i * cols + j] += y * (dy - dot);
                }
            }
        }
    };
    
    return result;
}

} // namespace functional
} // namespace nn
} // namespace mtf
