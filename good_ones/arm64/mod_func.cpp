#include <stdio.h>
#include "hocdec.h"
extern int nrnmpi_myid;
extern int nrn_nobanner_;
#if defined(__cplusplus)
extern "C" {
#endif

extern void _ode_neuron_reg(void);
extern void _probabilistic_syn_reg(void);

void modl_reg() {
  if (!nrn_nobanner_) if (nrnmpi_myid < 1) {
    fprintf(stderr, "Additional mechanisms from files\n");
    fprintf(stderr, " \"ode_neuron.mod\"");
    fprintf(stderr, " \"probabilistic_syn.mod\"");
    fprintf(stderr, "\n");
  }
  _ode_neuron_reg();
  _probabilistic_syn_reg();
}

#if defined(__cplusplus)
}
#endif
