#ifndef __PIM_TRANSACTION_H
#define __PIM_TRANSACTION_H

#include <stdint.h>

namespace dramsim3 {

enum class PimTransactionKind {
    START_COMPUTATION,
    LOAD_DATAFLOW_CONFIG,
    LOAD_WORKLOAD_CONFIG
};

struct DecodedPimTransaction {
    PimTransactionKind kind;
    uint64_t launch_mask = 0;
    int vcuts = 1;
    int hcuts = 1;
    int mcf = 1;
    int ucf = 1;
    int df = 0;
    int m_tile_size = 1;
    int vcuts_next = 1;
    int hcuts_next = 1;
    int kernel_size = 0;
    int stride = 0;
    int cut_no = 0;
    int load_type = 0;
    int dim_value = 0;
    uint64_t base_row = 0;
};

class PimTransactionDecoder {
   public:
    static DecodedPimTransaction Decode(uint64_t address);
};

}  // namespace dramsim3
#endif
