#ifndef DRAMSIM3_PIM_KERNEL_CONTEXT_H
#define DRAMSIM3_PIM_KERNEL_CONTEXT_H

#include <stdexcept>
#include "hbnpu/pim_config.h"
#include "hbnpu/pim_execution_state.h"

namespace dramsim3 {

// Values are captured at launch; consumers own a const copy.
struct PimKernelContext {
    PimDataflowConfig dataflow;
    int m = 0, n = 0, k = 0;
    uint64_t base_row_in = 0, base_row_weight = 0, base_row_output = 0;

    static PimKernelContext Capture(const PimDataflowConfig& config,
                                    const PimExecutionState& state) {
        PimKernelContext result;
        result.dataflow = config;
        result.m = state.m; result.n = state.n; result.k = state.k;
        result.base_row_in = state.base_row_in;
        result.base_row_weight = state.base_row_weight;
        result.base_row_output = state.base_row_output;
        return result;
    }

    void Validate(int channels) const {
        if (!dataflow.configured) throw std::invalid_argument("PIM context: configured");
        if (m <= 0) throw std::invalid_argument("PIM context: M");
        if (n <= 0) throw std::invalid_argument("PIM context: N");
        if (k <= 0) throw std::invalid_argument("PIM context: K");
        if (channels <= 0) throw std::invalid_argument("PIM context: channels");
        if (dataflow.m_tile_size <= 0) throw std::invalid_argument("PIM context: M tile");
        if (dataflow.mc <= 0) throw std::invalid_argument("PIM context: mc");
        if (dataflow.df != 0 && dataflow.df != 1)
            throw std::invalid_argument("PIM context: dataflow");
        if (dataflow.df == 1 && (dataflow.m_tile_size / 128) * dataflow.mcf <= 0)
            throw std::invalid_argument("PIM context: LH output tile");
    }
};

struct PimComputeBlock {
    int m_origin = 0, n_origin = 0, k_origin = 0;
    int m_extent = 0, n_extent = 0, k_extent = 0;
    bool first_k = false, final_k = false, last_block = false;
};

// Output coordinates use the legacy dataflow-specific transformed layout.
struct PimOutputBlock {
    int m_origin = 0, n_tile = 0, m_extent = 0;
    bool last_output = false;
};

}  // namespace dramsim3
#endif
