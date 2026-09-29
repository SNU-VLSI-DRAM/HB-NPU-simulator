#ifndef DRAMSIM3_PIM_TILE_EXECUTOR_H
#define DRAMSIM3_PIM_TILE_EXECUTOR_H
#include <functional>
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
    bool computation_finished = false;
    bool compute_exhausted = false, output_exhausted = false;
};
struct PimDescriptorSource {
    std::function<bool(PimComputeBlock&)> next_compute;
    std::function<bool(PimOutputBlock&)> next_output;
};
class PimTileExecutor {
   public:
    PimTileExecutor(const Config&, const PimKernelContext&);
    PimTickResult Tick(uint64_t clk, const PimTickFeedback&, const PimDescriptorSource&);
    bool CanAcceptCompute() const;
    void LoadCompute(const PimComputeBlock&);
    bool CanAcceptOutput() const;
    void LoadOutput(const PimOutputBlock&);
    PimExecutionState Snapshot() const { return state_; }
   private:
    friend class PimTileExecutorTestPeer;
    const Config& config_;
    const PimKernelContext context_;
    const PimDataflowConfig& pim_config_;
    PimExecutionState state_;
    PimComputeBlock compute_;
    PimOutputBlock output_;
    bool compute_active_ = false, output_active_ = false;
    bool compute_exhausted_ = false, output_exhausted_ = false;
};
}  // namespace dramsim3
#endif
