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


#endif //SLMC_FINAL_NEURALNETWORK_H
