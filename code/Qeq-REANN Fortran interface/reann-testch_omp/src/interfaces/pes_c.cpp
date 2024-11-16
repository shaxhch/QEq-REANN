#include <iostream>
#include "pes_cpp.hpp"
#include "pes_c.h"

using namespace std;

chPES* create_chpes(double *chcell, int *chpbc, int chnumatoms, char **chspecies, int chmaxnumtype, char **chatomtype, double *chmass) {
  return new chPes(chcell, chpbc, chnumatoms, chspecies, chmaxnumtype, chatomtype, chmass);
}

void delete_chpes(chPES* chpes){
  delete chpes;
}

void pes_reann_chout(chPES* chpes, int chnumatoms, double *chcart, double *chenergy, double *chforce) {
  chpes->reann_chout(chnumatoms, chcart, chenergy, chforce);
  return;
}
