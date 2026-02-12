#include "nn/layers.hpp"
#include "core/ops_cpu.hpp"
#include <cmath>

namespace mtf {
namespace nn {

Dense::Dense(size_t input_dim, size_t output_dim, bool use_bias) 
    : use_bias_(use_bias) {
    init_parameters(input_dim, output_dim);
}

void Dense::init_parameters(size_t input_dim, size_t output_dim) {
    float std = std::sqrt(2.0f / (input_dim + output_dim));
    
    core::Tensor w_data({input_dim, output_dim});
    w_data.randn(0.0f, std);
    weight_ = autograd::Node::create(w_data, true, "Dense_W");

    if (use_bias_) {
        core::Tensor b_data({1, output_dim});
        b_data.fill(0.0f);
        bias_ = autograd::Node::create(b_data, true, "Dense_b");
    }
}

autograd::NodePtr Dense::forward(autograd::NodePtr input) {
    auto output = autograd::matmul(input, weight_);
    if (use_bias_) {
        output = output + bias_; 
    }
    return output;
}

std::vector<autograd::NodePtr> Dense::parameters() const {
    std::vector<autograd::NodePtr> params = {weight_};
    if (use_bias_) {
        params.push_back(bias_);
    }
    return params;
}

} // namespace nn
} // namespace mtf
