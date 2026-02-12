#include "autograd/engine.hpp"
#include <stack>
#include <unordered_set>
#include <algorithm>

namespace mtf {
namespace autograd {

void Engine::backward(NodePtr root) {
    if (!root) return;

    std::vector<NodePtr> sorted = topological_sort(root);

    if (root->grad.size() == 0) {
        // Uninitialized gradient handling can be added here if needed
    }
    root->grad.fill(1.0f);

    for (auto it = sorted.rbegin(); it != sorted.rend(); ++it) {
        NodePtr node = *it;
        if (node->backward_fn) {
            node->backward_fn();
        }
    }
}

std::vector<NodePtr> Engine::topological_sort(NodePtr root) {
    std::vector<NodePtr> sorted;
    std::unordered_set<NodePtr> visited;
    
    std::function<void(NodePtr)> visit = [&](NodePtr node) {
        if (!node || visited.count(node)) return;
        visited.insert(node);
        
        for (auto& parent : node->parents) {
            visit(parent);
        }
        sorted.push_back(node);
    };

    visit(root);
    return sorted;
}

} // namespace autograd
} // namespace mtf
