#ifndef DRAMSIM3_LEGACY_TILE_SCHEDULER_H
#define DRAMSIM3_LEGACY_TILE_SCHEDULER_H

#include "pim_kernel_context.h"

namespace dramsim3 {

// Selects blocks only: no bank readiness, stage counters, or simulated ticks.
class LegacyTileScheduler {
   public:
    LegacyTileScheduler(const PimKernelContext& context, int channels);
    bool NextCompute(PimComputeBlock& block);
    bool NextOutput(PimOutputBlock& block);
    void ProjectProgress(PimExecutionState&, bool compute_exhausted,
                         bool output_exhausted) const;

   private:
    const PimKernelContext context_;
    const int k_tile_size_;
    int m_origin_ = 0, n_origin_ = 0, k_origin_ = 0;
    int output_m_origin_ = 0, output_n_tile_ = 0;
};

}  // namespace dramsim3
#endif
