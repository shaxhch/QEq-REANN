#pragma once
#include <torch/torch.h>
#include <torch/script.h> 
#include <vector>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <string>

class chPes {
public:
    chPes(double *chcell, int *chpbc, int chnumatoms, char **chspecies, int chmaxnumtype, char **chatomtype, double *chmass);
    ~chPes();
    void reann_chout(int chnumatoms, double *chcart, double *chenergy, double *chforce);
  double *chcell;int *chpbc;long int chnumatoms;char **chspecies; int chmaxnumtype;char **chatomtype; double *chmass;
  torch::Dtype tensor_type;
  torch::DeviceType tensor_device_type;
  torch::Tensor tensor_device;
  torch::jit::script::Module chpes_model;

private:
  torch::Tensor chpbc_t;
  torch::Tensor chcell_t;
  torch::Tensor chspecies_t;
  torch::Tensor chmass_t;
  std::vector<int> arr_chpbc;
  std::vector<double> arr_chcell;
  std::vector<int> arr_chspecies;
  std::vector<double> arr_chmass;
};
