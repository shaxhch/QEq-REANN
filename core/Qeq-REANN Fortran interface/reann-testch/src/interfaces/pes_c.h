#ifdef __cplusplus
extern "C" {
  class chPes;
  typedef chPes chPES;
#else
  typedef struct chPES chPES;
#endif

chPES* create_chpes(double *chcell, int *chpbc, int chnumatoms, char **chspecies, int chmaxnumtype, char **chatomtype, double *chmass);

void delete_chpes(chPES* chpes);

void pes_reann_chout(chPES* chpes, int chnumatoms, double *chcart, double *chenergy, double *chforce);
#ifdef __cplusplus
}
#endif
