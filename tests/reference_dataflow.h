#ifndef HBNPU_TEST_REFERENCE_DATAFLOW_H
#define HBNPU_TEST_REFERENCE_DATAFLOW_H
#include "hbnpu/pim_config.h"
#include "reference_pim_transaction.h"
namespace dramsim3 {
// Decode historical fixture constants only in the test harness.
inline void LoadReferenceDataflow(PimDataflowConfig& config,
                                  const DecodedPimTransaction& decoded) {
    config.mcf = decoded.mcf;
    config.ucf = decoded.ucf;
    config.df = decoded.df;
    config.mc = decoded.mcf * decoded.ucf;
    config.m_tile_size = decoded.m_tile_size;
    config.kernel_size = decoded.kernel_size;
    config.stride = decoded.stride;
    config.configured = true;
}
}
#endif
