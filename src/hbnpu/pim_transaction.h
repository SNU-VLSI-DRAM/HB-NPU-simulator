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
    int mcf = 1;
    int ucf = 1;
    int df = 0;
    int m_tile_size = 1;
    int kernel_size = 0;
    int stride = 0;
    int load_type = 0;
    int dim_value = 0;
    uint64_t base_row = 0;
};

class PimTransactionDecoder {
   public:
    // Single-array wire format (see docs/pim-trace-format.md):
    // bit 0: launch; otherwise bits 1..2: dimension selector or config (3).
    // Config: mcf[3:5], ucf[6:8], df[9], tile[10:13], kernel[14:18], stride[19:23].
    // Workload: dimension[3:34], base row[35:56]. No partition fields.
    static DecodedPimTransaction Decode(uint64_t address);
};

}  // namespace dramsim3
#endif
