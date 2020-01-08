//
// Created by fraktal on 05.01.2020.
//

#ifndef SLMC_FINAL_NEURALNETWORK_H
#define SLMC_FINAL_NEURALNETWORK_H

#include <iostream>
#include <torch/torch.h>
#include <armadillo>

struct NetImpl : torch::nn::Module {
    NetImpl()
            : conv1(torch::nn::Conv2dOptions(1, 64, /*kernel_size=*/3).padding(1)),
              conv2(torch::nn::Conv2dOptions(64, 64, /*kernel_size=*/3).padding(1)),
              fc1(256, 16),
              fc2(16, 1){
        register_module("conv1", conv1);
        register_module("conv2", conv2);
        register_module("fc1", fc1);
        register_module("fc2", fc2);
    }

    torch::Tensor forward(torch::Tensor x) {
        x = torch::selu(torch::max_pool2d(conv1->forward(x), 2));
        x = torch::selu(
                torch::max_pool2d(conv2->forward(x), 2));
        x = x.view({-1, 256});
        x = torch::selu(fc1->forward(x));
        x = fc2->forward(x);
        return x;
    }

    torch::nn::Conv2d conv1;
    torch::nn::Conv2d conv2;
    torch::nn::Linear fc1;
    torch::nn::Linear fc2;


};
TORCH_MODULE(Net); // creates module holder for NetImpl


void trainNetwork(Net model, int epochs, int batchSize, const string filename_x, const string filename_y, torch::DeviceType device_type, bool save_mode=false) {
    auto data_set = MyDataset(filename_x, filename_y, true, false).map(torch::data::transforms::Stack<>());
    auto data_loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
            std::move(data_set), batchSize);

    cout << "training data loaded" << endl;

    torch::optim::Adam optimizer(model->parameters(), torch::optim::AdamOptions(1e-3));
    model->train();

    size_t batch_idx = 0;
    float value_loss;
    for (int i = 0; i < epochs; i++) {
        cout << i << endl;
        for (auto &batch : *data_loader) {
            auto data = batch.data.to(device_type), targets = batch.target.to(device_type);
            optimizer.zero_grad();
            auto output = model->forward(data);
            auto loss = torch::mse_loss(output.squeeze(), targets);
            value_loss = loss.template item<float>();
            AT_ASSERT(!std::isnan(loss.template item<float>()));
            loss.backward();
            optimizer.step();
        }


        printf("\rTrain Epoch: %d Loss: %.4f",
               i, value_loss);
    }
    if(save_mode) torch::save(model, "modelCNN.pt");

}


#endif //SLMC_FINAL_NEURALNETWORK_H
