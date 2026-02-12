#pragma once

#include <string>
#include <vector>

namespace mtf {
namespace nn {

struct ModelMetadata {
    std::string model_name;
    std::string description;
    size_t input_dim;
    std::string input_description;
    std::string input_example;
    std::string input_format;
    size_t output_dim;
    std::string output_description;
    std::vector<std::string> layer_paths;
    std::vector<std::string> activations;
    
    bool save(const std::string& filepath) const;
    static ModelMetadata load(const std::string& filepath);
};

} // namespace nn
} // namespace mtf
