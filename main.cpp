#include <iostream>
#include <torch/torch.h>
#include <armadillo>
#include "dataload.h"
#include "neuralnetwork.h"
#include "slmc.h"

using namespace std;
using namespace arma;


int main() {
    // 1. SETTING THE DEVICE

    // a) we choose CPU or GPU training
    torch::DeviceType device_type;
    if (torch::cuda::is_available()) {
        std::cout << "CUDA available! Training on GPU." << std::endl;
        device_type = torch::kCUDA;
    } else {
        std::cout << "Training on CPU." << std::endl;
        device_type = torch::kCPU;
    }
    torch::Device device(device_type);

    torch::manual_seed(1);

    // b) definition of the neural network

    Net modelCNN;
    modelCNN->to(device);



    // c) parameters of the neural network

    bool mode_pretrained = false;

    if(mode_pretrained) {

        const int64_t kTrainBatchSize = 256;
        const int64_t epochs = 300;


        //2. DATA LOAD

        const string X_all = "new_data.txt"; // configurations - training
        const string y_all = "new_label.txt"; // energies of the configurations - training

        const string X_test = "conf_test.txt"; // configurations - testing
        const string y_test = "energy_test.txt"; // energies of the configurations - testing

        auto data_set = MyDataset(X_all, y_all, true, false).map(torch::data::transforms::Stack<>());
        auto data_loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
                std::move(data_set), kTrainBatchSize);

        cout << "training data loaded" << endl;

        //3. TRAINING

        torch::optim::Adam optimizer(modelCNN->parameters(), torch::optim::AdamOptions(1e-3));
        modelCNN->train();

        size_t batch_idx = 0;
        float value_loss;
        for (int i = 0; i < epochs; i++) {
            cout << i << endl;
            for (auto &batch : *data_loader) {
                auto data = batch.data.to(device), targets = batch.target.to(device);
                optimizer.zero_grad();
                auto output = modelCNN->forward(data);
                auto loss = torch::mse_loss(output.squeeze(), targets);
                value_loss = loss.template item<float>();
                AT_ASSERT(!std::isnan(loss.template item<float>()));
                loss.backward();
                optimizer.step();
            }


            printf("\rTrain Epoch: %d Loss: %.4f",
                   i, value_loss);
        }

        torch::save(modelCNN, "modelCNN.pt");
    }

    else {
        torch::load(modelCNN, "modelCNN.pt");
    }



    return 0;
}