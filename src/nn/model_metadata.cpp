#include "nn/model_metadata.hpp"
#include <fstream>
#include <sstream>
#include <iostream>

namespace mtf {
namespace nn {

bool ModelMetadata::save(const std::string& filepath) const {
    std::ofstream file(filepath);
    if (!file.is_open()) {
        return false;
    }
    
    file << "model_name=" << model_name << std::endl;
    file << "description=" << description << std::endl;
    file << "input_dim=" << input_dim << std::endl;
    file << "input_description=" << input_description << std::endl;
    file << "input_example=" << input_example << std::endl;
    file << "input_format=" << input_format << std::endl;
    file << "output_dim=" << output_dim << std::endl;
    file << "output_description=" << output_description << std::endl;
    
    file << "layer_count=" << layer_paths.size() << std::endl;
    for (size_t i = 0; i < layer_paths.size(); ++i) {
        file << "layer_" << i << "=" << layer_paths[i] << std::endl;
    }
    
    file << "activation_count=" << activations.size() << std::endl;
    for (size_t i = 0; i < activations.size(); ++i) {
        file << "activation_" << i << "=" << activations[i] << std::endl;
    }
    
    file.close();
    return true;
}

ModelMetadata ModelMetadata::load(const std::string& filepath) {
    ModelMetadata metadata;
    
    try {
        std::ifstream file(filepath);
        
        if (!file.is_open()) {
            std::cerr << "Error: Cannot open metadata file: " << filepath << std::endl;
            return metadata;
        }
        
        std::string line;
        while (std::getline(file, line)) {
            if (line.empty()) continue;
            
            size_t pos = line.find('=');
            if (pos == std::string::npos) continue;
            
            std::string key = line.substr(0, pos);
            std::string value = line.substr(pos + 1);
            
            while (!key.empty() && (key.back() == ' ' || key.back() == '\r' || key.back() == '\n')) {
                key.pop_back();
            }
            while (!value.empty() && (value.back() == ' ' || value.back() == '\r' || value.back() == '\n')) {
                value.pop_back();
            }
            
            if (key == "model_name") {
                metadata.model_name = value;
            } else if (key == "description") {
                metadata.description = value;
            } else if (key == "input_dim") {
                metadata.input_dim = std::stoull(value);
            } else if (key == "input_description") {
                metadata.input_description = value;
            } else if (key == "input_example") {
                metadata.input_example = value;
            } else if (key == "input_format") {
                metadata.input_format = value;
            } else if (key == "output_dim") {
                metadata.output_dim = std::stoull(value);
            } else if (key == "output_description") {
                metadata.output_description = value;
            } else if (key == "layer_count") {
                size_t count = std::stoull(value);
                metadata.layer_paths.resize(count);
            } else if (key.find("layer_") == 0) {
                size_t idx = std::stoull(key.substr(6));
                if (idx < metadata.layer_paths.size()) {
                    metadata.layer_paths[idx] = value;
                }
            } else if (key == "activation_count") {
                size_t count = std::stoull(value);
                metadata.activations.resize(count);
            } else if (key.find("activation_") == 0) {
                size_t idx = std::stoull(key.substr(11));
                if (idx < metadata.activations.size()) {
                    metadata.activations[idx] = value;
                }
            }
        }
        
        file.close();
    } catch (const std::exception& e) {
        std::cerr << "Exception in ModelMetadata::load: " << e.what() << std::endl;
    }
    
    return metadata;
}

} // namespace nn
} // namespace mtf
