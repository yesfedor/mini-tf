#pragma once

#include "core/memory.hpp"
#include "core/tensor.hpp"
#include "core/ops_cpu.hpp"

#include "autograd/node.hpp"
#include "autograd/engine.hpp"

#include "nn/layers.hpp"
#include "nn/activations.hpp"
#include "nn/loss.hpp"
#include "nn/model_metadata.hpp"

#include "optim/optimizer.hpp"
#include "optim/sgd.hpp"
#include "optim/adam.hpp"

namespace mtf {

inline autograd::NodePtr Variable(core::Tensor::Shape shape, bool requires_grad = false) {
    core::Tensor t(shape);
    t.randn(0.0f, 0.1f);
    return autograd::Node::create(t, requires_grad, "Variable");
}

inline autograd::NodePtr Variable(core::Tensor t, bool requires_grad = false) {
    return autograd::Node::create(t, requires_grad, "Variable");
}

inline autograd::NodePtr matmul(const autograd::NodePtr& a, const autograd::NodePtr& b) {
    return autograd::matmul(a, b);
}

} // namespace mtf
