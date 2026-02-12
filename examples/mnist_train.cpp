#include "mini_tf.hpp"
#include <iostream>
#include <vector>
#include <fstream>
#include <iomanip>

void train_mnist() {
    std::cout << "Starting MNIST training..." << std::endl;

    size_t batch_size = 32;
    size_t input_dim = 784;
    size_t hidden_dim = 128;
    size_t output_dim = 10;
    float learning_rate = 0.001f;
    int epochs = 5;

    mtf::nn::Dense fc1(input_dim, hidden_dim);
    mtf::nn::Dense fc2(hidden_dim, output_dim);

    std::vector<mtf::autograd::NodePtr> params = fc1.parameters();
    std::vector<mtf::autograd::NodePtr> params2 = fc2.parameters();
    params.insert(params.end(), params2.begin(), params2.end());

    mtf::optim::Adam optimizer(params, learning_rate);
    mtf::nn::CrossEntropyLoss criterion;

    std::cout << "Training on dummy data (checking flow)..." << std::endl;

    for (int epoch = 0; epoch < epochs; ++epoch) {
        float total_loss = 0.0f;
        int steps = 100;

        for (int step = 0; step < steps; ++step) {
            mtf::core::Tensor x_data({batch_size, input_dim});
            // Normalize random data to be more realistic (0-1 range roughly)
            x_data.randn(0.0f, 0.5f); 
            // Clamp to avoid large values which might saturate sigmoid/exp
            for(size_t i=0; i<x_data.size(); ++i) x_data[i] = std::abs(x_data[i]); 
            
            auto x = mtf::Variable(x_data, false);

            mtf::core::Tensor y_data({batch_size, output_dim});
            y_data.fill(0.0f);
            
            for(size_t i=0; i<batch_size; ++i) {
                int label = rand() % 10;
                y_data[{i, static_cast<size_t>(label)}] = 1.0f;
            }
            auto y = mtf::Variable(y_data, false);

            auto h1 = fc1(x);
            auto a1 = mtf::nn::functional::relu(h1);
            auto logits = fc2(a1);
            auto probs = mtf::nn::functional::softmax(logits);
            
            auto loss = criterion(probs, y);

            optimizer.zero_grad();
            mtf::autograd::Engine::backward(loss);
            optimizer.step();

            total_loss += loss->value[0];
            
            if (std::isnan(loss->value[0])) {
                std::cerr << "NaN Loss detected at step " << step << std::endl;
                return;
            }
        }
        
        std::cout << "Epoch " << epoch + 1 << ", Loss: " << total_loss / steps << std::endl;
    }
}

int main() {
    train_mnist();
    return 0;
}
