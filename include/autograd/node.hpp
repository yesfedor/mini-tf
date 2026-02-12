#pragma once

#include <memory>
#include <vector>
#include <functional>
#include <string>
#include <iostream>

#include "core/tensor.hpp"

namespace mtf {
namespace autograd {

class Node;
using NodePtr = std::shared_ptr<Node>;

class Node : public std::enable_shared_from_this<Node> {
public:
    core::Tensor value;
    core::Tensor grad;
    
    std::vector<NodePtr> parents;
    std::string op_name;
    
    using BackwardFn = std::function<void()>;
    BackwardFn backward_fn;
    
    bool requires_grad;

    Node(core::Tensor val, bool req_grad = false, std::string op = "");
    
    void zero_grad();

    static NodePtr create(core::Tensor val, bool req_grad = false, std::string op = "");
};

NodePtr operator+(const NodePtr& a, const NodePtr& b);
NodePtr operator-(const NodePtr& a, const NodePtr& b);
NodePtr operator*(const NodePtr& a, const NodePtr& b);
NodePtr matmul(const NodePtr& a, const NodePtr& b);

} // namespace autograd
} // namespace mtf
