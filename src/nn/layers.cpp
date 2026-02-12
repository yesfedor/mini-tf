#include "nn/layers.hpp"
#include "core/ops_cpu.hpp"
#include <cmath>
#include <fstream>
#include <sstream>

namespace mtf {
namespace nn {

Dense::Dense(size_t input_dim, size_t output_dim, bool use_bias) 
    : use_bias_(use_bias), input_dim_(input_dim), output_dim_(output_dim) {
    init_parameters(input_dim, output_dim);
}

Dense::Dense(size_t input_dim, size_t output_dim, bool use_bias,
             const core::Tensor& weight, const core::Tensor& bias)
    : use_bias_(use_bias), input_dim_(input_dim), output_dim_(output_dim) {
    weight_ = autograd::Node::create(weight, true, "Dense_W");
    if (use_bias_) {
        bias_ = autograd::Node::create(bias, true, "Dense_b");
    }
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

bool Dense::save(const std::string& filepath) const {
    std::string weight_path = filepath + "_weight.bin";
    std::string bias_path = filepath + "_bias.bin";
    std::string meta_path = filepath + "_meta.txt";
    
    if (!weight_->value.save(weight_path)) {
        return false;
    }
    
    if (use_bias_ && !bias_->value.save(bias_path)) {
        return false;
    }
    
    std::ofstream meta(meta_path);
    if (!meta.is_open()) {
        return false;
    }
    meta << input_dim_ << " " << output_dim_ << " " << (use_bias_ ? 1 : 0) << std::endl;
    meta.close();
    
    return true;
}

Dense Dense::load(const std::string& filepath) {
    std::string weight_path = filepath + "_weight.bin";
    std::string bias_path = filepath + "_bias.bin";
    std::string meta_path = filepath + "_meta.txt";
    
    std::ifstream meta(meta_path);
    if (!meta.is_open()) {
        std::cerr << "Error: Cannot open meta file: " << meta_path << std::endl;
        return Dense(1, 1);
    }
    
    size_t input_dim, output_dim;
    int use_bias;
    meta >> input_dim >> output_dim >> use_bias;
    meta.close();
    
    auto weight = core::Tensor::load(weight_path);
    if (weight.size() == 0) {
        std::cerr << "Error: Cannot load weight from: " << weight_path << std::endl;
        return Dense(1, 1);
    }
    
    core::Tensor bias;
    if (use_bias) {
        bias = core::Tensor::load(bias_path);
        if (bias.size() == 0) {
            std::cerr << "Error: Cannot load bias from: " << bias_path << std::endl;
            return Dense(1, 1);
        }
    } else {
        bias = core::Tensor({1, output_dim});
        bias.fill(0.0f);
    }
    
    return Dense(input_dim, output_dim, use_bias != 0, weight, bias);
}

} // namespace nn
} // namespace mtf
