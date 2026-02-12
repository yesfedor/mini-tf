#pragma once

#include <vector>
#include <memory>
#include <string>
#include <initializer_list>
#include <iostream>

namespace mtf {
namespace core {

class Tensor {
public:
    using Shape = std::vector<size_t>;
    using Strides = std::vector<size_t>;

    Tensor();
    Tensor(const Shape& shape);
    Tensor(const Shape& shape, const std::vector<float>& data);
    Tensor(std::initializer_list<float> data, const Shape& shape);
    
    Tensor(const Tensor& other);
    Tensor(Tensor&& other) noexcept;
    Tensor& operator=(const Tensor& other);
    Tensor& operator=(Tensor&& other) noexcept;

    ~Tensor();

    float& at(const std::vector<size_t>& indices);
    const float& at(const std::vector<size_t>& indices) const;
    
    float& operator[](size_t index);
    const float& operator[](size_t index) const;

    float& operator[](std::initializer_list<size_t> indices);
    const float& operator[](std::initializer_list<size_t> indices) const;

    const Shape& shape() const { return shape_; }
    const Strides& strides() const { return strides_; }
    size_t size() const { return size_; }
    float* data() { return data_; }
    const float* data() const { return data_; }

    void fill(float value);
    void randn(float mean = 0.0f, float std = 1.0f);
    void print() const;
    
    bool save(const std::string& filepath) const;
    static Tensor load(const std::string& filepath);
    
    static Strides compute_strides(const Shape& shape);

private:
    float* data_;
    size_t size_;
    Shape shape_;
    Strides strides_;
    bool owns_memory_;
};

} // namespace core
} // namespace mtf
