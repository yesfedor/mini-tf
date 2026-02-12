#include "mini_tf.hpp"
#include <iostream>
#include <vector>
#include <sstream>
#include <string>
#include <algorithm>
#include <fstream>

std::vector<float> parse_input(const std::string& line, size_t expected_dim, const std::string& format) {
    std::vector<float> result;
    std::istringstream iss(line);
    float val;
    
    while (iss >> val && result.size() < expected_dim) {
        if (format.find("binary") != std::string::npos) {
            if (val != 0.0f && val != 1.0f) {
                return {};
            }
        }
        result.push_back(val);
    }
    
    return result.size() == expected_dim ? result : std::vector<float>();
}

void print_prediction(const mtf::autograd::NodePtr& output, const mtf::nn::ModelMetadata& metadata) {
    if (metadata.output_dim == 1) {
        float value = output->value[{0, 0}];
        
        if (!metadata.activations.empty() && metadata.activations.back() == "sigmoid") {
            int prediction = (value > 0.5f) ? 1 : 0;
            std::cout << "Prediction: " << value << " (" << prediction << ")" << std::endl;
        } else {
            std::cout << "Prediction: " << value << std::endl;
        }
    } else {
        std::cout << "Predictions: ";
        for (size_t i = 0; i < metadata.output_dim; ++i) {
            std::cout << output->value[{0, i}];
            if (i < metadata.output_dim - 1) std::cout << ", ";
        }
        std::cout << std::endl;
        
        if (metadata.activations.back() == "softmax") {
            float max_val = output->value[{0, 0}];
            size_t max_idx = 0;
            for (size_t i = 1; i < metadata.output_dim; ++i) {
                if (output->value[{0, i}] > max_val) {
                    max_val = output->value[{0, i}];
                    max_idx = i;
                }
            }
            std::cout << "Class: " << max_idx << " (probability: " << max_val << ")" << std::endl;
        }
    }
}

mtf::autograd::NodePtr forward_pass(std::vector<mtf::nn::Dense>& layers, 
                                     const std::vector<std::string>& activations,
                                     mtf::autograd::NodePtr input) {
    auto x = input;
    
    for (size_t i = 0; i < layers.size(); ++i) {
        x = layers[i](x);
        
        if (i < activations.size() && !activations[i].empty()) {
            const std::string& act = activations[i];
            if (act == "relu") {
                x = mtf::nn::functional::relu(x);
            } else if (act == "tanh") {
                x = mtf::nn::functional::tanh(x);
            } else if (act == "sigmoid") {
                x = mtf::nn::functional::sigmoid(x);
            } else if (act == "softmax") {
                x = mtf::nn::functional::softmax(x);
            }
        }
    }
    
    return x;
}

int main(int argc, char* argv[]) {
    std::string model_path;
    
    if (argc > 1) {
        model_path = argv[1];
    } else {
        std::cout << "Enter model path (e.g., models/xor): ";
        std::getline(std::cin, model_path);
    }
    
    if (model_path.empty()) {
        std::cerr << "Error: Model path cannot be empty" << std::endl;
        return 1;
    }
    
    std::string metadata_path = model_path + "_metadata.txt";
    
    std::ifstream test_file(metadata_path);
    if (!test_file.is_open()) {
        std::cerr << "Error: File not found: " << metadata_path << std::endl;
        return 1;
    }
    test_file.close();
    
    auto metadata = mtf::nn::ModelMetadata::load(metadata_path);
    
    if (metadata.model_name.empty()) {
        std::cerr << "Error: Cannot load model metadata from: " << metadata_path << std::endl;
        std::cerr << "Make sure the model was saved with metadata." << std::endl;
        return 1;
    }
    
    std::vector<mtf::nn::Dense> layers;
    try {
        for (size_t i = 0; i < metadata.layer_paths.size(); ++i) {
            layers.push_back(mtf::nn::Dense::load(metadata.layer_paths[i]));
        }
    } catch (const std::exception& e) {
        std::cerr << "Exception while loading layers: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "Unknown exception while loading layers" << std::endl;
        return 1;
    }
    
    if (layers.empty()) {
        std::cerr << "Error: No layers loaded!" << std::endl;
        return 1;
    }
    std::cout << "\n" << metadata.input_description << std::endl;
    std::cout << "Example: " << metadata.input_example << std::endl;
    std::cout << "Type 'quit' to exit\n" << std::endl;
    
    std::string input_line;
    while (true) {
        std::cout << "Input: ";
        std::getline(std::cin, input_line);
        
        if (input_line == "quit" || input_line == "exit" || input_line == "q") {
            break;
        }
        
        auto input_data = parse_input(input_line, metadata.input_dim, metadata.input_format);
        
        if (input_data.empty()) {
            std::cerr << "Error: Invalid input. Expected " << metadata.input_dim 
                      << " values in format: " << metadata.input_format << std::endl;
            continue;
        }
        
        try {
            auto x_tensor = mtf::core::Tensor({1, metadata.input_dim}, input_data);
            auto x = mtf::Variable(x_tensor, false);
            
            auto output = forward_pass(layers, metadata.activations, x);
            
            std::cout << "Input: ";
            for (size_t i = 0; i < input_data.size(); ++i) {
                std::cout << input_data[i];
                if (i < input_data.size() - 1) std::cout << " ";
            }
            std::cout << std::endl;
            
            print_prediction(output, metadata);
            std::cout << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "Exception during prediction: " << e.what() << std::endl;
        } catch (...) {
            std::cerr << "Unknown exception during prediction" << std::endl;
        }
    }
    
    std::cout << "Goodbye!" << std::endl;
    return 0;
}
