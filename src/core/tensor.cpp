#include "core/tensor.hpp"
#include "core/memory.hpp"
#include <numeric>
#include <algorithm>
#include <random>
#include <iostream>
#include <cstring>
#include <fstream>

namespace mtf {
namespace core {

Tensor::Tensor() : data_(nullptr), size_(0), shape_({}), strides_({}), owns_memory_(false) {}

Tensor::Tensor(const Shape& shape) : shape_(shape), owns_memory_(true) {
    size_ = 1;
    for (auto dim : shape) {
        size_ *= dim;
    }
    strides_ = compute_strides(shape);
    data_ = static_cast<float*>(aligned_alloc(size_ * sizeof(float)));
}

Tensor::Tensor(const Shape& shape, const std::vector<float>& data) : Tensor(shape) {
    if (data.size() != size_) {
        std::cerr << "Error: Tensor data size mismatch" << std::endl;
        return;
    }
    std::memcpy(data_, data.data(), size_ * sizeof(float));
}

Tensor::Tensor(std::initializer_list<float> data, const Shape& shape) : Tensor(shape) {
    if (data.size() != size_) {
        std::cerr << "Error: Tensor data size mismatch" << std::endl;
        return;
    }
    std::copy(data.begin(), data.end(), data_);
}

Tensor::Tensor(const Tensor& other) : 
    size_(other.size_), shape_(other.shape_), strides_(other.strides_), owns_memory_(true) {
    if (other.data_) {
        data_ = static_cast<float*>(aligned_alloc(size_ * sizeof(float)));
        std::memcpy(data_, other.data_, size_ * sizeof(float));
    } else {
        data_ = nullptr;
    }
}

Tensor::Tensor(Tensor&& other) noexcept :
    data_(other.data_), size_(other.size_), shape_(std::move(other.shape_)), strides_(std::move(other.strides_)), owns_memory_(other.owns_memory_) {
    other.data_ = nullptr;
    other.size_ = 0;
    other.owns_memory_ = false;
}

Tensor& Tensor::operator=(const Tensor& other) {
    if (this == &other) return *this;

    if (owns_memory_ && data_) {
        aligned_free(data_);
    }

    size_ = other.size_;
    shape_ = other.shape_;
    strides_ = other.strides_;
    owns_memory_ = true;

    if (other.data_) {
        data_ = static_cast<float*>(aligned_alloc(size_ * sizeof(float)));
        std::memcpy(data_, other.data_, size_ * sizeof(float));
    } else {
        data_ = nullptr;
    }
    return *this;
}

Tensor& Tensor::operator=(Tensor&& other) noexcept {
    if (this == &other) return *this;

    if (owns_memory_ && data_) {
        aligned_free(data_);
    }

    data_ = other.data_;
    size_ = other.size_;
    shape_ = std::move(other.shape_);
    strides_ = std::move(other.strides_);
    owns_memory_ = other.owns_memory_;

    other.data_ = nullptr;
    other.size_ = 0;
    other.owns_memory_ = false;
    return *this;
}

Tensor::~Tensor() {
    if (owns_memory_ && data_) {
        aligned_free(data_);
    }
}

Tensor::Strides Tensor::compute_strides(const Shape& shape) {
    Strides s(shape.size());
    size_t stride = 1;
    for (int i = static_cast<int>(shape.size()) - 1; i >= 0; --i) {
        s[i] = stride;
        stride *= shape[i];
    }
    return s;
}

float& Tensor::at(const std::vector<size_t>& indices) {
    size_t offset = 0;
    for (size_t i = 0; i < indices.size(); ++i) {
        offset += indices[i] * strides_[i];
    }
    return data_[offset];
}

const float& Tensor::at(const std::vector<size_t>& indices) const {
    size_t offset = 0;
    for (size_t i = 0; i < indices.size(); ++i) {
        offset += indices[i] * strides_[i];
    }
    return data_[offset];
}

float& Tensor::operator[](size_t index) {
    return data_[index];
}

const float& Tensor::operator[](size_t index) const {
    return data_[index];
}

float& Tensor::operator[](std::initializer_list<size_t> indices) {
    size_t offset = 0;
    size_t i = 0;
    for (auto index : indices) {
        offset += index * strides_[i];
        i++;
    }
    return data_[offset];
}

const float& Tensor::operator[](std::initializer_list<size_t> indices) const {
    size_t offset = 0;
    size_t i = 0;
    for (auto index : indices) {
        offset += index * strides_[i];
        i++;
    }
    return data_[offset];
}

void Tensor::fill(float value) {
    for (size_t i = 0; i < size_; ++i) {
        data_[i] = value;
    }
}

void Tensor::randn(float mean, float std) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> d(mean, std);
    for (size_t i = 0; i < size_; ++i) {
        data_[i] = d(gen);
    }
}

void Tensor::print() const {
    std::cout << "Tensor shape=(";
    for (size_t i = 0; i < shape_.size(); ++i) {
        std::cout << shape_[i] << (i < shape_.size() - 1 ? ", " : "");
    }
    std::cout << ")" << std::endl;
    
    if (size_ <= 100) {
        for (size_t i = 0; i < size_; ++i) {
            std::cout << data_[i] << " ";
            if ((i + 1) % (shape_.back()) == 0) std::cout << std::endl;
        }
    } else {
        std::cout << "[ ... " << size_ << " elements ... ]" << std::endl;
    }
}

bool Tensor::save(const std::string& filepath) const {
    std::ofstream file(filepath, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open file for writing: " << filepath << std::endl;
        return false;
    }
    
    size_t shape_size = shape_.size();
    file.write(reinterpret_cast<const char*>(&shape_size), sizeof(size_t));
    file.write(reinterpret_cast<const char*>(shape_.data()), shape_size * sizeof(size_t));
    file.write(reinterpret_cast<const char*>(&size_), sizeof(size_t));
    file.write(reinterpret_cast<const char*>(data_), size_ * sizeof(float));
    
    file.close();
    return true;
}

Tensor Tensor::load(const std::string& filepath) {
    std::ifstream file(filepath, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open file for reading: " << filepath << std::endl;
        return Tensor();
    }
    
    size_t shape_size;
    file.read(reinterpret_cast<char*>(&shape_size), sizeof(size_t));
    
    Shape shape(shape_size);
    file.read(reinterpret_cast<char*>(shape.data()), shape_size * sizeof(size_t));
    
    size_t size;
    file.read(reinterpret_cast<char*>(&size), sizeof(size_t));
    
    Tensor result(shape);
    file.read(reinterpret_cast<char*>(result.data_), size * sizeof(float));
    
    file.close();
    return result;
}

} // namespace core
} // namespace mtf
