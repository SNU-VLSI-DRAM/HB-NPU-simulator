#ifndef DRAMSIM3_PIM_INSTRUCTION_SEQUENCER_H
#define DRAMSIM3_PIM_INSTRUCTION_SEQUENCER_H
#include <memory>

#include "hbnpu/pim_instruction_buffer.h"
#include "hbnpu/pim_tile_executor.h"
namespace dramsim3 {
struct PimSequencerStats {
    uint64_t dispatched_instructions = 0, retired_instructions = 0,
             compute_resource_stall_cycles = 0, load_resource_stall_cycles = 0,
             output_resource_stall_cycles = 0,
             instruction_starvation_cycles = 0, backend_drain_cycle = 0,
             end_cycle = 0;
    bool backend_drained = false, end_completed = false;
};
class PimInstructionSequencer {
   public:
    explicit PimInstructionSequencer(const Config& config) : config_(config) {}
    PimTickResult Tick(uint64_t cycle, PimInstructionBuffer&,
                       const PimTickFeedback&);
    bool Complete() const { return stats_.end_completed; }
    PimExecutionState Snapshot() const {
        return executor_ ? executor_->Snapshot() : PimExecutionState{};
    }
    const PimSequencerStats& Stats() const { return stats_; }

   private:
    void Dispatch(uint64_t cycle, PimInstructionBuffer&);
    const Config& config_;
    std::unique_ptr<const PimKernelContext> context_;
    std::unique_ptr<PimTileExecutor> executor_;
    PimInstructionOpcode expected_ = PimInstructionOpcode::CONFIG;
    uint64_t group_ = 0, groups_ = 0, n_groups_ = 0, output_n_groups_ = 0;
    int expected_k_ = 0;
    bool compute_active_ = false, output_active_ = false;
    bool load_active_ = false;
    uint32_t compute_sequence_ = 0, output_sequence_ = 0;
    PimSequencerStats stats_;
};
}  // namespace dramsim3
#endif
