#include "core/ops_cpu.hpp"
#include <cmath>
#include <algorithm>
#include <cassert>

namespace mtf {
namespace core {
namespace ops {

Tensor add(const Tensor& a, const Tensor& b) {
    Tensor result(a.shape());
    for (size_t i = 0; i < a.size(); ++i) {
        result[i] = a[i] + b[i];
    }
    return result;
}

Tensor sub(const Tensor& a, const Tensor& b) {
    Tensor result(a.shape());
    for (size_t i = 0; i < a.size(); ++i) {
        result[i] = a[i] - b[i];
    }
    return result;
}

Tensor mul(const Tensor& a, const Tensor& b) {
    Tensor result(a.shape());
    for (size_t i = 0; i < a.size(); ++i) {
        result[i] = a[i] * b[i];
    }
    return result;
}

Tensor div(const Tensor& a, const Tensor& b) {
    Tensor result(a.shape());
    for (size_t i = 0; i < a.size(); ++i) {
        result[i] = a[i] / b[i];
    }
    return result;
}

Tensor add_scalar(const Tensor& a, float scalar) {
    Tensor result(a.shape());
    for (size_t i = 0; i < a.size(); ++i) {
        result[i] = a[i] + scalar;
    }
    return result;
}

Tensor mul_scalar(const Tensor& a, float scalar) {
    Tensor result(a.shape());
    for (size_t i = 0; i < a.size(); ++i) {
        result[i] = a[i] * scalar;
    }
    return result;
}

Tensor matmul(const Tensor& a, const Tensor& b) {
    size_t M = a.shape()[0];
    size_t K = a.shape()[1];
    size_t N = b.shape()[1];

    assert(a.shape()[1] == b.shape()[0]);

    Tensor result({M, N});
    result.fill(0.0f);

    const float* A_ptr = a.data();
    const float* B_ptr = b.data();
    float* C_ptr = result.data();

    for (size_t i = 0; i < M; ++i) {
        for (size_t k = 0; k < K; ++k) {
            float val_a = A_ptr[i * K + k];
            for (size_t j = 0; j < N; ++j) {
                C_ptr[i * N + j] += val_a * B_ptr[k * N + j];
            }
        }
    }
    return result;
}

Tensor transpose(const Tensor& a) {
    size_t rows = a.shape()[0];
    size_t cols = a.shape()[1];
    Tensor result({cols, rows});
    
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            result[j * rows + i] = a[i * cols + j];
        }
    }
    return result;
}

Tensor sum(const Tensor& a) {
    float s = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) {
        s += a[i];
    }
    Tensor result({1});
    result[0] = s;
    return result;
}

Tensor mean(const Tensor& a) {
    Tensor s = sum(a);
    s[0] /= static_cast<float>(a.size());
    return s;
}

Tensor relu(const Tensor& a) {
    Tensor result(a.shape());
    for (size_t i = 0; i < a.size(); ++i) {
        result[i] = std::max(0.0f, a[i]);
    }
    return result;
}

Tensor sigmoid(const Tensor& a) {
    Tensor result(a.shape());
    for (size_t i = 0; i < a.size(); ++i) {
        result[i] = 1.0f / (1.0f + std::exp(-a[i]));
    }
    return result;
}

Tensor exp(const Tensor& a) {
    Tensor result(a.shape());
    for (size_t i = 0; i < a.size(); ++i) {
        result[i] = std::exp(a[i]);
    }
    return result;
}

Tensor log(const Tensor& a) {
    Tensor result(a.shape());
    for (size_t i = 0; i < a.size(); ++i) {
        result[i] = std::log(a[i] + 1e-8f);
    }
    return result;
}

Tensor max(const Tensor& a, const Tensor& b) {
    Tensor result(a.shape());
    for (size_t i = 0; i < a.size(); ++i) {
        result[i] = std::max(a[i], b[i]);
    }
    return result;
}

} // namespace ops
} // namespace core
} // namespace mtf
