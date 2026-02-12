#include "mini_tf.hpp"
#include <iostream>
#include <vector>
#include <cmath>
#include <sstream>
#include <string>

void load_and_test() {
    std::cout << "Loading XOR model..." << std::endl;
    
    auto fc1 = mtf::nn::Dense::load("models/xor_fc1");
    auto fc2 = mtf::nn::Dense::load("models/xor_fc2");
    
    std::cout << "Model loaded successfully!" << std::endl;
    std::cout << "\nEnter 4-bit input (4 numbers 0 or 1, separated by spaces):" << std::endl;
    std::cout << "Example: 1 0 1 1" << std::endl;
    std::cout << "Type 'quit' to exit\n" << std::endl;
    
    std::string input_line;
    while (true) {
        std::cout << "Input: ";
        std::getline(std::cin, input_line);
        
        if (input_line == "quit" || input_line == "exit" || input_line == "q") {
            break;
        }
        
        std::vector<float> x_raw;
        std::istringstream iss(input_line);
        float val;
        int count = 0;
        
        while (iss >> val && count < 4) {
            if (val != 0.0f && val != 1.0f) {
                std::cerr << "Error: Input must be 0 or 1" << std::endl;
                break;
            }
            x_raw.push_back(val);
            count++;
        }
        
        if (x_raw.size() != 4) {
            std::cerr << "Error: Please enter exactly 4 values (0 or 1)" << std::endl;
            continue;
        }
        
        int sum = static_cast<int>(x_raw[0]) + static_cast<int>(x_raw[1]) + 
                  static_cast<int>(x_raw[2]) + static_cast<int>(x_raw[3]);
        int expected = (sum % 2 == 1) ? 1 : 0;
        
        auto x_tensor = mtf::core::Tensor({1, 4}, x_raw);
        auto x = mtf::Variable(x_tensor, false);
        
        auto h1 = fc1(x);
        auto a1 = mtf::nn::functional::tanh(h1);
        auto logits = fc2(a1);
        auto preds = mtf::nn::functional::sigmoid(logits);
        
        float p = preds->value[{0, 0}];
        int prediction = (p > 0.5f) ? 1 : 0;
        
        std::cout << "Input: " << x_raw[0] << " " << x_raw[1] << " " 
                  << x_raw[2] << " " << x_raw[3] << std::endl;
        std::cout << "Sum: " << sum << " (";
        if (sum % 2 == 1) {
            std::cout << "odd";
        } else {
            std::cout << "even";
        }
        std::cout << ")" << std::endl;
        std::cout << "Expected: " << expected << std::endl;
        std::cout << "Prediction: " << p << " (" << prediction << ")" << std::endl;
        std::cout << "Result: " << (prediction == expected ? "✓ CORRECT" : "✗ WRONG") << std::endl;
        std::cout << std::endl;
    }
    
    std::cout << "Goodbye!" << std::endl;
}

int main() {
    load_and_test();
    return 0;
}
