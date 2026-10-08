#ifndef DRAMSIM3_PIM_TILE_EXECUTOR_H
#define DRAMSIM3_PIM_TILE_EXECUTOR_H
#include <functional>
#include "configuration.h"
#include "hbnpu/pim_kernel_context.h"
#include "hbnpu/pim_command_batch.h"
#include "hbnpu/pim_row_operation.h"

namespace dramsim3 {
struct PimTickFeedback {
    bool wait_refresh = false, in_refresh = false;
    bool weight_pending = false, input_pending = false, output_pending = false;
    int row_vectors_completed = 0;
};
enum class PimProgress {
    ResidentVectorIssued, ResidentTileReady, InputVectorIssued, BlockStreamingExhausted,
    OutputStageReleased, OutputBlockIssued, PipelineDrained
};
struct PimTickResult {
    PimCommandBatch batch;
    PimRowOperation row;
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
    void LoadResident(const PimComputeBlock&);
    void StartCompute(const PimComputeBlock&);
    bool CanStartCompute() const { return compute_active_ && !compute_requested_; }
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
    bool compute_requested_ = false;
    int row_vectors_ = 0, row_observed_ = 0;
    bool compute_exhausted_ = false, output_exhausted_ = false;
};
}  // namespace dramsim3
#endif
