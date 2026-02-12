#pragma once

#include "node.hpp"
#include <vector>

namespace mtf {
namespace autograd {

class Engine {
public:
    static void backward(NodePtr root);
    static std::vector<NodePtr> topological_sort(NodePtr root);
};

} // namespace autograd
} // namespace mtf
