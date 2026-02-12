#pragma once

#include <vector>
#include <string>
#include "autograd/node.hpp"

namespace mtf {
namespace nn {

class Layer {
public:
    virtual ~Layer() = default;

    virtual autograd::NodePtr forward(autograd::NodePtr input) = 0;
    virtual std::vector<autograd::NodePtr> parameters() const = 0;
    
    autograd::NodePtr operator()(autograd::NodePtr input) {
        return forward(input);
    }
};

class Dense : public Layer {
public:
    Dense(size_t input_dim, size_t output_dim, bool use_bias = true);
    Dense(size_t input_dim, size_t output_dim, bool use_bias, 
          const core::Tensor& weight, const core::Tensor& bias);

    autograd::NodePtr forward(autograd::NodePtr input) override;
    std::vector<autograd::NodePtr> parameters() const override;
    
    bool save(const std::string& filepath) const;
    static Dense load(const std::string& filepath);

private:
    autograd::NodePtr weight_;
    autograd::NodePtr bias_;
    bool use_bias_;
    size_t input_dim_;
    size_t output_dim_;
    
    void init_parameters(size_t input_dim, size_t output_dim);
};

} // namespace nn
} // namespace mtf
