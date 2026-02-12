#include "autograd/node.hpp"
#include "core/ops_cpu.hpp"
#include <iostream>

namespace mtf {
namespace autograd {

Node::Node(core::Tensor val, bool req_grad, std::string op)
    : value(val), op_name(op), requires_grad(req_grad) {
    if (requires_grad) {
        grad = core::Tensor(val.shape());
        grad.fill(0.0f);
    }
}

NodePtr Node::create(core::Tensor val, bool req_grad, std::string op) {
    return std::make_shared<Node>(val, req_grad, op);
}

void Node::zero_grad() {
    if (requires_grad) {
        grad.fill(0.0f);
    }
}

NodePtr operator+(const NodePtr& a, const NodePtr& b) {
    auto result = Node::create(core::ops::add(a->value, b->value), 
                               a->requires_grad || b->requires_grad, 
                               "Add");
    result->parents = {a, b};

    result->backward_fn = [result, a, b]() {
        if (a->requires_grad) {
            a->grad = core::ops::add(a->grad, result->grad);
        }
        if (b->requires_grad) {
            if (b->value.shape().size() == 2 && result->grad.shape().size() == 2 &&
                b->value.shape()[0] == 1 && result->grad.shape()[0] > 1 &&
                b->value.shape()[1] == result->grad.shape()[1]) {
                size_t M = result->grad.shape()[0];
                size_t N = result->grad.shape()[1];
                const float* grad_ptr = result->grad.data();
                float* b_grad_ptr = b->grad.data();
                
                for (size_t j = 0; j < N; ++j) {
                    float sum = 0.0f;
                    for (size_t i = 0; i < M; ++i) {
                        sum += grad_ptr[i * N + j];
                    }
                    b_grad_ptr[j] += sum;
                }
            } else {
                b->grad = core::ops::add(b->grad, result->grad);
            }
        }
    };
    return result;
}

NodePtr operator-(const NodePtr& a, const NodePtr& b) {
    auto result = Node::create(core::ops::sub(a->value, b->value), 
                               a->requires_grad || b->requires_grad, 
                               "Sub");
    result->parents = {a, b};

    result->backward_fn = [result, a, b]() {
        if (a->requires_grad) {
            a->grad = core::ops::add(a->grad, result->grad);
        }
        if (b->requires_grad) {
            auto neg_grad = core::ops::mul_scalar(result->grad, -1.0f);
            b->grad = core::ops::add(b->grad, neg_grad);
        }
    };
    return result;
}

NodePtr operator*(const NodePtr& a, const NodePtr& b) {
    auto result = Node::create(core::ops::mul(a->value, b->value), 
                               a->requires_grad || b->requires_grad, 
                               "Mul");
    result->parents = {a, b};

    result->backward_fn = [result, a, b]() {
        if (a->requires_grad) {
            auto da = core::ops::mul(result->grad, b->value);
            a->grad = core::ops::add(a->grad, da);
        }
        if (b->requires_grad) {
            auto db = core::ops::mul(result->grad, a->value);
            b->grad = core::ops::add(b->grad, db);
        }
    };
    return result;
}

NodePtr matmul(const NodePtr& a, const NodePtr& b) {
    auto result = Node::create(core::ops::matmul(a->value, b->value), 
                               a->requires_grad || b->requires_grad, 
                               "MatMul");
    result->parents = {a, b};

    result->backward_fn = [result, a, b]() {
        if (a->requires_grad) {
            auto b_t = core::ops::transpose(b->value);
            auto da = core::ops::matmul(result->grad, b_t);
            a->grad = core::ops::add(a->grad, da);
        }
        if (b->requires_grad) {
            auto a_t = core::ops::transpose(a->value);
            auto db = core::ops::matmul(a_t, result->grad);
            b->grad = core::ops::add(b->grad, db);
        }
    };
    return result;
}

} // namespace autograd
} // namespace mtf
