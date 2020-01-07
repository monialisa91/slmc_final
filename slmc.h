//
// Created by fraktal on 07.01.2020.
//

#ifndef SLMC_FINAL_SLMC_H
#define SLMC_FINAL_SLMC_H

#include <iostream>
#include <torch/torch.h>
#include "dataload.h"
#include "neuralnetwork.h"


mat Swap_sites(mat lattice) {
    double r;
    int lattice_n1, lattice_n2;
    int size = lattice.n_rows;
    double swap_var;

    do {
        lattice_n1 = (rand() % static_cast<int>(size -1 + 1));
    } while(lattice(lattice_n1, lattice_n1) < 0.00001);

    do {
        lattice_n2 = (rand() % static_cast<int>(size -1 + 1));

    } while(lattice(lattice_n2, lattice_n2) > 0.1);


    swap_var = lattice(lattice_n1, lattice_n1);
    lattice(lattice_n1, lattice_n1) = lattice(lattice_n2, lattice_n2);
    lattice(lattice_n2, lattice_n2) = swap_var;

    return lattice;

}

double energy_conf(mat hamiltonian, double beta, double cp) {
    mat eigvec;
    vec eigval;
    eig_sym(eigval, eigvec, hamiltonian);
    int N = hamiltonian.n_rows;
    double E = 0;
    double T = 1.0/beta;
    for(int j=0; j<N; j++){
        E += log(1+exp(-beta*(eigval(j)-cp)));
    }
    return -T*E;
}

inline mat MC(mat initial_hamiltonian, int MC_steps, double beta, double cp) {
    int acc = 0;
    double E0, E_new, delta, r;
    mat new_hamiltonian;
    E0 = energy_conf(initial_hamiltonian, beta, cp);
    for (int i = 0; i < MC_steps; i++) {
        new_hamiltonian = Swap_sites(initial_hamiltonian);
        E_new = energy_conf(new_hamiltonian, beta, cp);
        delta = E_new - E0;
        if (delta < 0) {
            initial_hamiltonian = new_hamiltonian;
            E0 = E_new;
            acc++;
        }

        else {
            r = ((double) rand() / (RAND_MAX));
            if (exp(-delta * beta) >= r) {
                initial_hamiltonian = new_hamiltonian;
                E0 = E_new;
                acc++;
            }
        }
    }
    //cout << "acc= " << acc << endl;
    return initial_hamiltonian;

}


// SLMC

mat confToHam (rowvec conf, double U, int t, int L) {
    int size_matrix = conf.n_cols;
    mat lattice(size_matrix, size_matrix);
    lattice.zeros();

    for(int i=0; i<size_matrix; i++) lattice(i, i) = conf(i)*U;

    for(int i = 0; i<size_matrix-L; i++) {
        lattice(i, i+L) = -t;
        lattice(i+L, i) = -t;
    }

    for(int i=0; i<L; i++) {
        lattice(i, i+size_matrix-L) = -t;
        lattice(i+size_matrix-L, i) = -t;
    }

    for(int i=0; i<=L-1; i++){
        lattice(i*L, i*L+L-1) = -t;
        lattice(i*L+L-1,  i*L) = -t;
    }

    for(int i=1; i<=L-1; i++) {
        for(int j=0; j<=L-1; j++) {
            lattice(i+j*L-1, i+j*L) = -t;
            lattice(i+j*L, i+j*L-1) = -t;
        }
    }
    return lattice;
}

rowvec HamToConf(mat hamiltonian, double U) {
    int L = hamiltonian.n_rows;
    rowvec conf(L);
    for(int i=0; i<L; i++) {
        conf(i) = hamiltonian(i, i)/U;
    }

    return conf;

}

mat LastHam (mat data, double U, int t, int L) {

    int D = data.n_rows;
    rowvec LastConf = data.row(D-1);
    mat rslt = confToHam(LastConf, U, t, L);

    return rslt;
}

double energySLMC (mat hamiltonian, double U,  NetImpl modelCNN, torch::DeviceType device_type, bool transform) {
    rowvec conf = HamToConf(hamiltonian, U);
    torch::Tensor conf2 = RowVecToTensor(conf);
    conf2 = conf2.reshape({1, 1, 10, 10});
    conf2 = conf2.to(device_type);
    auto output2 = modelCNN.forward(conf2);
    float rslt = output2.template item<float>();
    if(transform == true)
        rslt = -exp(rslt);
    return (double) rslt;

}

mat SLMC_effective (mat initial_hamiltonian, double U, double beta, int MC_steps, NetImpl modelCNN, torch::DeviceType device_type, bool transform) {
    double cp = U/2.0;
    double E0 = energySLMC(initial_hamiltonian, U,  modelCNN, device_type, transform);
    double E0real = energy_conf(initial_hamiltonian, beta, cp);
//    cout << "\n energy_init " << E0 << " " << E0real << endl;
    double E_new, delta, r;
    mat new_hamiltonian;
    int acc = 0;
    int acc_en = 0;

    for(int i=0; i<MC_steps; i++) {
        new_hamiltonian = Swap_sites(initial_hamiltonian);
        E_new = energySLMC(new_hamiltonian, U, modelCNN, device_type, transform);
        delta = E_new - E0;
        if(delta < 0) {
            initial_hamiltonian = new_hamiltonian;
            E0 = E_new;
            acc++;
            acc_en++;
        }
        else {
            r = ((double) rand() / (RAND_MAX));
            if(exp(-beta*delta) >= r) {
                initial_hamiltonian = new_hamiltonian;
                E0 = E_new;
                acc++;
            }
        }
    }
    double Enewreal = energy_conf(initial_hamiltonian, beta, U/2.0);
//    cout << "\n Energy end " << E0  << "Enew_real " << Enewreal << endl;
//    cout << "\n accepted:" << (double) acc/MC_steps <<endl;
//    cout << "\n accepted_en:" << (double) acc_en/MC_steps <<endl;


    return initial_hamiltonian;
}

mat SLMC(mat initial_hamiltonian, double U, double beta, double cp, int n_conf, int MC_steps, NetImpl modelCNN, torch::DeviceType device_type, bool transform) {
    double EA = energy_conf(initial_hamiltonian, beta, cp);
    double EA_eff = energySLMC(initial_hamiltonian, U,  modelCNN, device_type, transform);
    double EB, EB_eff, delta, r;
    mat new_hamiltonian;
    int acc =0;
    cout << "\nEA " << EA << endl;
    cout << "EAeff " << EA_eff << endl;

    for(int i=0; i<n_conf; i++) {
        new_hamiltonian = SLMC_effective(initial_hamiltonian, U, beta, MC_steps, modelCNN, device_type, transform);
        EB = energy_conf(new_hamiltonian, beta, cp);
        EB_eff = energySLMC(new_hamiltonian, U,  modelCNN, device_type, transform);
        cout << "EB " << EB << " " <<"EBeff " << EB_eff << endl;

        delta = EB - EA  + EA_eff - EB_eff;
        if(delta < 0) {
            initial_hamiltonian = new_hamiltonian;
            EA = EB;
            EA_eff = EB_eff;
            acc++;
        }
        else {
            r = ((double) rand() / (RAND_MAX));
            if(exp(-beta*delta) > r) {
                initial_hamiltonian = new_hamiltonian;
                EA = EB;
                EA_eff = EB_eff;
                acc++;
            }
        }
    }
    cout << "accepted all " << acc << endl;
    return initial_hamiltonian;
}


#endif //SLMC_FINAL_SLMC_H
