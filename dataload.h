//
// Created by fraktal on 05.01.2020.
//

#ifndef SLMC_FINAL_DATALOAD_H
#define SLMC_FINAL_DATALOAD_H

#include <iostream>
#include <torch/torch.h>
#include <armadillo>

using namespace std;
using namespace arma;


torch::Tensor VecToTensor(vec data) {
    int n = data.n_rows;
    torch::Tensor tensor = torch::ones({n});

    for(int i=0; i<n; i++) {
        tensor[i] = data(i);
    }
    cout << tensor.sizes() << endl;
    return tensor;
}

torch::Tensor RowVecToTensor(rowvec data) {
    int n = data.n_cols;
    torch::Tensor tensor = torch::ones({n});

    for(int i=0; i<n; i++) {
        tensor[i] = data(i);
    }
    return tensor;
}

torch::Tensor MatToTensor(mat data) {
    int n = data.n_rows;
    int m = data.n_cols;

    torch::Tensor tensor = torch::ones({n, m});

    for(int i=0; i<n; i++) {
        for(int j=0; j<m; j++) {
            tensor[i][j] = data(i, j);
        }
    }
    return tensor;
}


torch::Tensor mat_read_data(const std::string& loc, bool reshape)
{
    mat conf;
    conf.load(loc);
    torch::Tensor tensor = MatToTensor(conf);
    if(reshape == true){
        tensor = tensor.reshape({tensor.size(0), 1, 10, 10});
    }

    cout << tensor.size(0) << endl;

    return tensor;
};

torch::Tensor vec_read_data(const std::string& loc, bool transform) {
    vec energies;
    energies.load(loc);
    if(transform == true)
        energies.transform( [](double val) { return (log(abs(val))); } );
    torch::Tensor tensor = VecToTensor(energies);
    cout << tensor.size(0) << endl;

    return tensor;
};

class MyDataset : public torch::data::Dataset<MyDataset>
{
private:
    torch::Tensor states_, labels_;

public:
    explicit MyDataset(const std::string& loc_states, const std::string& loc_labels, bool reshape, bool transform)
            : states_(mat_read_data(loc_states, reshape)),
              labels_(vec_read_data(loc_labels, transform)) {   };

    torch::data::Example<> get(size_t index) override {
        return {states_[index], labels_[index]};

    }
    torch::optional<size_t> size() const override {
        return states_.sizes()[0];
    }

};

#endif //SLMC_FINAL_DATALOAD_H
