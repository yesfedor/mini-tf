#include "mini_tf.hpp"
#include <iostream>
#include <vector>
#include <cmath>

void train_xor() {
    std::cout << "Starting 4-bit XOR training (parity check)..." << std::endl;

    size_t input_dim = 4;
    size_t hidden_dim = 32; 
    size_t output_dim = 1; 
    float learning_rate = 0.01f; 
    int epochs = 5000;

    mtf::nn::Dense fc1(input_dim, hidden_dim);
    mtf::nn::Dense fc2(hidden_dim, output_dim);

    std::vector<mtf::autograd::NodePtr> params = fc1.parameters();
    std::vector<mtf::autograd::NodePtr> params2 = fc2.parameters();
    params.insert(params.end(), params2.begin(), params2.end());

    mtf::optim::Adam optimizer(params, learning_rate);
    mtf::nn::MSELoss criterion;

    std::vector<float> x_raw;
    std::vector<float> y_raw;
    
    for (int i = 0; i < 16; ++i) {
        int a = (i >> 3) & 1;
        int b = (i >> 2) & 1;
        int c = (i >> 1) & 1;
        int d = i & 1;
        int sum = a + b + c + d;
        int target = (sum % 2 == 1) ? 1 : 0;
        
        x_raw.push_back(static_cast<float>(a));
        x_raw.push_back(static_cast<float>(b));
        x_raw.push_back(static_cast<float>(c));
        x_raw.push_back(static_cast<float>(d));
        y_raw.push_back(static_cast<float>(target));
    }

    auto x_tensor = mtf::core::Tensor({16, 4}, x_raw);
    auto y_tensor = mtf::core::Tensor({16, 1}, y_raw);

    auto x = mtf::Variable(x_tensor, false);
    auto y = mtf::Variable(y_tensor, false);

    for (int epoch = 0; epoch <= epochs; ++epoch) {
        auto h1 = fc1(x);
        auto a1 = mtf::nn::functional::tanh(h1); 
        auto logits = fc2(a1);
        auto preds = mtf::nn::functional::sigmoid(logits);
        
        auto loss = criterion(preds, y);

        optimizer.zero_grad();
        mtf::autograd::Engine::backward(loss);
        optimizer.step();

        if (epoch % 500 == 0) {
            std::cout << "Epoch " << epoch << ", Loss: " << loss->value[0] << std::endl;
        }
    }

    std::cout << "\nTraining complete. Testing..." << std::endl;
    
    auto h1_test = fc1(x);
    auto a1_test = mtf::nn::functional::tanh(h1_test);
    auto logits_test = fc2(a1_test);
    auto preds_test = mtf::nn::functional::sigmoid(logits_test);
    
    std::cout << "\nInput (A B C D) | Sum | Expected | Prediction" << std::endl;
    std::cout << "-----------------------------------------------" << std::endl;
    for (size_t i = 0; i < 16; ++i) {
        int a = static_cast<int>(x_raw[i*4]);
        int b = static_cast<int>(x_raw[i*4+1]);
        int c = static_cast<int>(x_raw[i*4+2]);
        int d = static_cast<int>(x_raw[i*4+3]);
        int sum = a + b + c + d;
        float p = preds_test->value[{i, 0}];
        std::cout << a << " " << b << " " << c << " " << d 
                  << " | " << sum << " | " << static_cast<int>(y_raw[i]) 
                  << " | " << p << " (" << (p > 0.5f ? "1" : "0") << ")" << std::endl;
    }
    
    std::cout << "\nSaving model..." << std::endl;
    if (fc1.save("models/xor_fc1")) {
        std::cout << "Saved fc1 to models/xor_fc1" << std::endl;
    } else {
        std::cerr << "Failed to save fc1" << std::endl;
    }
    
    if (fc2.save("models/xor_fc2")) {
        std::cout << "Saved fc2 to models/xor_fc2" << std::endl;
    } else {
        std::cerr << "Failed to save fc2" << std::endl;
    }
}

int main() {
    train_xor();
    return 0;
}
