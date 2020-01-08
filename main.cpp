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

        const string X_all = "new_data.txt"; // configurations - training
        const string y_all = "new_label.txt"; // energies of the configurations - training

        const string X_test = "conf_test.txt"; // configurations - testing
        const string y_test = "energy_test.txt"; // energies of the configurations - testing

        trainNetwork(modelCNN, epochs, kTrainBatchSize, X_all, y_all, device_type, true);
    }

    else {
        torch::load(modelCNN, "modelCNN.pt");
    }

    //4. Iterative SLMC

    // a) setting the parameters of the model

    srand(time(NULL));

    double U = 4.0;
    double cp = U/2.0;
    double beta = 1.0/0.2;
    double t = 1;
    int L = 10;

    mat configurations(10000, L*L);
    configurations.load( "conf_test.txt");
    mat new_ham;

    mat init_ham = HamConf(configurations, U, t, L, 1000);

    // b) setting the parameters of the additional training

    const int64_t kTrainBatchSize_slmc = 10;
    const int64_t epochs_slmc = 20;

    const string X_add = "conf_add.txt"; // add_configurations - training
    const string y_add = "energy_add.txt"; // add_energies of the configurations - training

    for(int j=0; j<1; j++) {
        new_ham = SLMC(init_ham, U, beta, cp, 100, 100, modelCNN, device_type, false, X_add, y_add);
        cout << "additional configurations added " << endl;
//        trainNetwork(modelCNN, 20, 8, X_add, y_add, device_type);

    }

    return 0;
}