#ifndef DRAMSIM3_PIM_TILE_EXECUTOR_H
#define DRAMSIM3_PIM_TILE_EXECUTOR_H
#include "configuration.h"
#include "pim_kernel_context.h"
#include "pim_command_batch.h"

namespace dramsim3 {
struct PimTickFeedback {
    bool wait_refresh = false, in_refresh = false;
    bool weight_pending = false, input_pending = false, output_pending = false;
};
enum class PimProgress {
    ResidentVectorIssued, InputVectorIssued, BlockStreamingExhausted,
    OutputStageReleased, OutputBlockIssued, PipelineDrained
};
struct PimTickResult {
    PimCommandBatch batch;
    std::vector<PimProgress> events;
    bool finished = false;
    bool compute_exhausted = false, output_exhausted = false;
};
class PimTileExecutor {
   public:
    PimTileExecutor(const Config&, const PimKernelContext&);
    PimTickResult RunLegacy(uint64_t clk, const PimTickFeedback&, PimExecutionState&);
   private:
    const Config& config_;
    const PimKernelContext context_;
    const PimDataflowConfig& pim_config_;
};
}  // namespace dramsim3
#endif
