#include "pes_cpp.hpp"
using namespace std;
using namespace torch::indexing;

int select_gpu();
template<typename T>
std::vector<T> trans_new_arr(T* arr,int length) {
  std::vector<T> arr_new;
  for(int i=0;i<length;i++){
    arr_new.push_back(arr[i]);
  }
  return arr_new;
}

//---------------------------------------------------------
chPes::chPes(double *_chcell, int *_chpbc, int _chnumatoms, char **_chspecies, int _chmaxnumtype, char **_chatomtype, double *_chmass):
chcell(_chcell), chpbc(_chpbc), chnumatoms(_chnumatoms), chspecies(_chspecies), chmaxnumtype(_chmaxnumtype), chatomtype(_chatomtype),chmass(_chmass){

  //set model precision
  std::string datatype="double"; //float

  tensor_device_type=torch::kCPU;
  tensor_device=torch::empty(1);

  //--------------------- 
  torch::jit::GraphOptimizerEnabledGuard guard{true};
  torch::jit::setGraphExecutorOptimize(true);
  if (datatype=="double") {
    tensor_type = torch::kDouble;
    chpes_model = torch::jit::load("REANN_CHA_DOUBLE.pt");
  }else{ 
    tensor_type=torch::kFloat;
    chpes_model = torch::jit::load("REANN_CHA_FLOAT.pt");
  }
  if (torch::cuda::is_available()) {
    int id=select_gpu();
    tensor_device_type=torch::kCUDA;
    auto device=torch::Device(tensor_device_type,id);
    tensor_device=tensor_device.to(device);
    chpes_model.to(tensor_device.device(),true);
    cout << "The simulations are performed on the GPU "<< id << endl;
  }
  chpes_model.eval();
  chpes_model=torch::jit::optimize_for_inference(chpes_model);

  //---------------------
  arr_chpbc=trans_new_arr(_chpbc,3);
  arr_chcell=trans_new_arr(_chcell,9);
  auto arr_chatom=trans_new_arr(_chspecies,_chnumatoms);
  auto arr_chatomtype=trans_new_arr(_chatomtype,_chmaxnumtype);
  arr_chmass=trans_new_arr(_chmass,_chnumatoms);
  this->chnumatoms=_chnumatoms;
  chpbc_t=torch::from_blob(arr_chpbc.data(),{3},torch::kInt).to(tensor_device.device(),true).contiguous();
  chcell_t=torch::from_blob(arr_chcell.data(),{3,3},torch::kDouble).to(tensor_type).to(tensor_device.device(),true).contiguous();
  std::unordered_map<std::string, int> chspecies_map;
  for(int i=0;i<chmaxnumtype;i++){
    chspecies_map[chatomtype[i]]=i;
  }
  arr_chspecies={};
  for(auto chatom: arr_chatom) {
    arr_chspecies.push_back(chspecies_map[chatom]);
  }
  chspecies_t=torch::from_blob(arr_chspecies.data(),{this->chnumatoms},torch::kInt).to(tensor_device.device(),true).contiguous();
  chmass_t=torch::from_blob(arr_chmass.data(),{this->chnumatoms},torch::kDouble).to(tensor_type).to(tensor_device.device(),true).contiguous();
}

//---------------------------------------------------------
chPes::~chPes(){
}
//---------------------------------------------------------
void chPes::reann_chout(int chnumatoms, double *chcart, double *chenergy, double *chforce) {
  auto arr_chxyz=trans_new_arr(chcart,this->chnumatoms*3);
  torch::Tensor chcart_t=torch::from_blob(arr_chxyz.data(),{this->chnumatoms,3},torch::kDouble).to(tensor_type).to(tensor_device.device(),true);
 auto outputs = chpes_model.forward({chpbc_t,chcart_t,chcell_t,chspecies_t,chmass_t}).toTuple()->elements();
  torch::Tensor chenergy_t=outputs[0].toTensor().to(torch::kDouble).cpu();
  torch::Tensor chforce_t=outputs[1].toTensor().to(torch::kDouble).cpu();

  auto chenergy_pd=chenergy_t.data_ptr<double>();
  for (int i=0;i<this->chnumatoms;i++){
    chenergy[i] = chenergy_pd[i];
  }
  for (int i=0;i<this->chnumatoms;i++){
    for (int j=0;j<this->chnumatoms;j++){
      for (int k=0;k<3;k++){
         double chforce_value = chforce_t[i][j][k].item<double>();
         chforce[i * this->chnumatoms * 3 + j * 3 + k] =  chforce_value;
      }
    }
  }
 return;
}
//-------------------------------------------------
int select_gpu() {
  int totalnodes, mynode;
  int trap_key = 0;
  system("nvidia-smi -q -d Memory |grep -A4 GPU|grep Used >gpu_info");
  ifstream gpu_sel("gpu_info");
  string texts;
  vector<double> memalloc;
  while (getline(gpu_sel,texts))
  {
    string tmp_str;
    stringstream allocation(texts);    
    allocation >> tmp_str;
    allocation >> tmp_str;
    allocation >> tmp_str;
    memalloc.push_back(std::stod(tmp_str));
  }
  gpu_sel.close();
  auto smallest=min_element(std::begin(memalloc),std::end(memalloc));
  auto id=distance(std::begin(memalloc), smallest);
  torch::Tensor tensor_device=torch::empty(1000,torch::Device(torch::kCUDA,id));
  system("rm gpu_info");
  return id;
}
